import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
from quadrotor import QuadRotor

class QuadRotorSlalom(QuadRotor):

    #environment to perfrom slalom navigation
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, num_gates=5, course_width=10, gate_spacing=15, gate_size=2.0):
        super().__init__()

        self.render_mode = render_mode

        # Env for slalom navigation
        self.num_gates = num_gates
        self.course_width = course_width
        self.gate_spacing = gate_spacing
        self.gate_size = gate_size
        self.gate_thickness = 0.2 

        # Gates 
        self.gates = []
        for i in range(num_gates):
            x_pos = (self.course_width / 2) * (-1)**i
            y_pos = 0 
            z_pos = (i + 1) * self.gate_spacing
            self.gates.append({'position': np.array([x_pos, y_pos, z_pos], dtype=np.float32), 'passed': False})

        # Observation Space 
        low_obs = np.array([-np.inf] * 16, dtype=np.float32)
        high_obs = np.array([np.inf] * 16, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)
        
        # State Variables
        self.quad_pos = np.zeros(3, dtype=np.float32)
        self.quad_vel = np.zeros(3, dtype=np.float32)
        self.quad_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # w,x,y,z
        self.quad_ang_vel = np.zeros(3, dtype=np.float32) # wx, wy, wz in body frame

        self.current_step = 0
        self.target_gate_index = 0
        self.trajectory = []

    def _get_obs(self):
        if self.target_gate_index >= len(self.gates):
            relative_pos_to_next_gate = np.zeros(3, dtype=np.float32)
        else:
            target_gate_pos = self.gates[self.target_gate_index]['position']
            relative_pos_to_next_gate = target_gate_pos - self.quad_pos
        
        return np.concatenate([
            self.quad_pos,
            self.quad_vel,
            self.quad_quat,
            self.quad_ang_vel,
            relative_pos_to_next_gate
        ]).astype(np.float32)

    def _get_info(self):
        dist_to_gate = np.inf
        if self.target_gate_index < len(self.gates):
            dist_to_gate = np.linalg.norm(self.gates[self.target_gate_index]['position'] - self.quad_pos)
        return {
            "target_gate": self.target_gate_index,
            "distance_to_gate": dist_to_gate,
            "position": self.quad_pos.copy(),
            "velocity": self.quad_vel.copy(),
            "orientation_quat": self.quad_quat.copy(),
            "angular_velocity_body": self.quad_ang_vel.copy()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.quad_pos = np.array([0, 0, 1.0], dtype=np.float32) + \
                        self.np_random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        self.quad_vel = np.zeros(3, dtype=np.float32)
        self.quad_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Start upright
        self.quad_ang_vel = np.zeros(3, dtype=np.float32)

        self.current_step = 0
        self.target_gate_index = 0
        for gate in self.gates:
            gate['passed'] = False

        self.trajectory = [self.quad_pos.copy()]
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        
        # Before vs After
        pos_before_step = self.quad_pos.copy()

        # Apply Physics 
        # Clip action to valid motor thrusts
        motor_thrusts = np.clip(action, self.action_space.low, self.action_space.high)
        
        f_fr, f_rl, f_fl, f_rr = motor_thrusts[0], motor_thrusts[1], motor_thrusts[2], motor_thrusts[3]

        # Roll torque (around body X, positive = right wing down)
        tau_x_body = self.L * (f_rl + f_fl - f_fr - f_rr) / 2.0 # Simplified for X
        # Pitch torque (around body Y, positive = nose up)
        tau_y_body = self.L * (f_rl + f_rr - f_fr - f_fl) / 2.0 # Simplified for X
        # Yaw torque (around body Z, positive = nose right)
        # (f_ccw - f_cw) * c_ratio -> ( (f_fl + f_rr) - (f_fr + f_rl) ) * c_ratio
        tau_z_body = self.c_torque_thrust_ratio * ( (motor_thrusts[2] + motor_thrusts[3]) - (motor_thrusts[0] + motor_thrusts[1]) )
        total_thrust = np.sum(motor_thrusts)
        torques_body = np.array([tau_x_body, tau_y_body, tau_z_body], dtype=np.float32)

        # Update angular velocity (Euler's rotation equation)
        ang_vel_dot_body = self.inv_I @ (torques_body - np.cross(self.quad_ang_vel, self.I @ self.quad_ang_vel))
        self.quad_ang_vel += ang_vel_dot_body * self.dt

        
        delta_quat_axis_angle = 0.5 * self.quad_ang_vel * self.dt
        delta_angle = np.linalg.norm(delta_quat_axis_angle)
        
        if delta_angle > 1e-8: # Avoid division by zero for small rotations
            axis = delta_quat_axis_angle / delta_angle
            delta_w = np.cos(delta_angle)
            q_dot = 0.5 * self.quaternion_multiply(np.concatenate(([0.0], self.quad_ang_vel)), self.quad_quat)
            self.quad_quat += q_dot * self.dt
            self.quad_quat /= np.linalg.norm(self.quad_quat) # Normalize quaternion


        R_body_to_world = self.quaternion_to_rotation_matrix(self.quad_quat)

        # Linear acceleration in world frame
        thrust_vector_body = np.array([0, 0, total_thrust], dtype=np.float32)
        acceleration_world = (R_body_to_world @ thrust_vector_body) / self.mass \
                             + np.array([0, 0, -self.gravity], dtype=np.float32)
        
        # Update linear velocity and position (Euler integration)
        self.quad_vel += acceleration_world * self.dt
        self.quad_pos += self.quad_vel * self.dt

        self.trajectory.append(self.quad_pos.copy())

        # Check Conditions and Calculate Reward
        terminated = False
        truncated = False
        reward = 0.0

        # Target gate info
        if self.target_gate_index < len(self.gates):
            target_gate = self.gates[self.target_gate_index]
            target_gate_pos = target_gate['position']
            dist_to_gate_before = np.linalg.norm(target_gate_pos - pos_before_step)
            dist_to_gate_after = np.linalg.norm(target_gate_pos - self.quad_pos)

            # Reward for getting closer to the target gate
            reward += (dist_to_gate_before - dist_to_gate_after) * 1.0

            # Check for passing the gate
            gate_z = target_gate_pos[2]
            # Crossed the Z-plane of the gate (either direction)
            if (pos_before_step[2] < gate_z <= self.quad_pos[2]) or \
               (self.quad_pos[2] < gate_z <= pos_before_step[2]):
                
                on_plane_pos = self.quad_pos 

                if abs(on_plane_pos[0] - target_gate_pos[0]) < self.gate_size / 2 and \
                   abs(on_plane_pos[1] - target_gate_pos[1]) < self.gate_size / 2:
                    reward += 100.0 # Big reward for passing
                    target_gate['passed'] = True
                    self.target_gate_index += 1
                else:
                    reward -= 50.0 # Passed Z-plane but missed opening
                    terminated = True
            
            reward -= dist_to_gate_after * 0.1 # Penalty for distance from gate center line

        # Check Collisions & Boundaries
        if self.quad_pos[2] < self.collision_radius: # Ground collision
            reward -= 100.0
            terminated = True

        for i, gate in enumerate(self.gates):
            if terminated: break
            # Only check unpassed or next target gate structure
            if not gate['passed'] or i == self.target_gate_index :
                gate_pos = gate['position']
                # If quad is close to the gate's Z-plane
                if abs(self.quad_pos[2] - gate_pos[2]) < self.gate_thickness / 2 + self.collision_radius:
                    # And if quad is outside the opening but within a larger bounding box of the gate structure
                    is_within_opening_x = abs(self.quad_pos[0] - gate_pos[0]) < self.gate_size / 2
                    is_within_opening_y = abs(self.quad_pos[1] - gate_pos[1]) < self.gate_size / 2
                    
                    # Approx extent of gate structure (e.g., gate_size + frame_width)
                    # Let's use gate_size for simplicity, assuming collision if near plane and not in opening
                    is_near_structure_x = abs(self.quad_pos[0] - gate_pos[0]) < self.gate_size 
                    is_near_structure_y = abs(self.quad_pos[1] - gate_pos[1]) < self.gate_size

                    if not (is_within_opening_x and is_within_opening_y) and \
                       (is_near_structure_x and is_near_structure_y):
                        reward -= 100.0
                        terminated = True
                        break
        
        # Out of bounds
        max_flight_extent = max(self.course_width * 2, (self.num_gates + 2) * self.gate_spacing)
        if np.any(np.abs(self.quad_pos[:2]) > max_flight_extent) or \
           self.quad_pos[2] > (self.num_gates + 2) * self.gate_spacing or \
           self.quad_pos[2] < -5: # flew too far up or down
            reward -= 100.0
            terminated = True

        # Check Termination/Truncation Conditions 
        if self.current_step >= self.max_steps_per_episode:
            truncated = True
        
        if self.target_gate_index >= len(self.gates): # All gates passed
            reward += 200.0 # Bonus for finishing
            truncated = True
        
        # if terminated, an additional penalty for early termination not due to success
        if terminated and not (self.target_gate_index >= len(self.gates)):
             reward -= 50 # Small penalty for any kind of crash


        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array' or self.render_mode == 'human':
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='.', markersize=2, linestyle='-', label='Quadrotor Path')
            if len(traj) > 0:
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, label='Start', depthshade=True)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, label='End', depthshade=True)

            for i, gate_info in enumerate(self.gates):
                pos = gate_info['position']
                half_size = self.gate_size / 2
                corners = np.array([
                    [pos[0] - half_size, pos[1] - half_size, pos[2]],
                    [pos[0] + half_size, pos[1] - half_size, pos[2]],
                    [pos[0] + half_size, pos[1] + half_size, pos[2]],
                    [pos[0] - half_size, pos[1] + half_size, pos[2]],
                    [pos[0] - half_size, pos[1] - half_size, pos[2]] # Close loop
                ])
                color = 'lime' if gate_info['passed'] else 'blue'
                ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=3, label=f'Gate {i+1}' if i==0 else None)

            max_x = self.course_width
            max_y = self.course_width / 2 + 2
            max_z = (self.num_gates + 1) * self.gate_spacing

            ax.set_xlim(-max_x, max_x)
            ax.set_ylim(-max_y, max_y)
            ax.set_zlim(0, max_z)
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Y position (m)")
            ax.set_zlabel("Z position (Height) (m)")
            ax.set_title(f"Quadrotor Slalom Trajectory (Gates: {self.target_gate_index}/{self.num_gates})")
            
            # Create a single legend for all unique labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.grid(True)

            if self.render_mode == 'human':
                plt.show()
            elif self.render_mode == 'rgb_array':
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img
        return None


    def close(self):
        plt.close('all')

#5 million timesteps is enough to train our quadrotor

if __name__ == "__main__":
    TRAIN_MODEL = False # set to true to train
    LOAD_MODEL_PATH = "quad_slalom_ppo.zip"
    SAVE_MODEL_PATH = "quad_slalom_ppo.zip"

    TOTAL_TIMESTEPS = 5000000 

    log_dir = "./quad_physics_slalom_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    print("Creating environment with detailed physics...")
    env = DummyVecEnv([lambda: QuadRotorSlalom(render_mode='rgb_array')])

    if TRAIN_MODEL:
        print(f"Training model for {TOTAL_TIMESTEPS} timesteps...")
        model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=3e-4, 
                    n_steps=2048,       
                    batch_size=64,     
                    gamma=0.99,        
                    gae_lambda=0.95,   
                    ent_coef=0.0,       
                   )
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        print(f"Saving model to {SAVE_MODEL_PATH}")
        model.save(SAVE_MODEL_PATH)
        del model 

    if not os.path.exists(LOAD_MODEL_PATH) and not TRAIN_MODEL:
        print(f"Error: Model file not found at {LOAD_MODEL_PATH}. Train the model first or set TRAIN_MODEL=True.")
    else:
        if os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading model from {LOAD_MODEL_PATH}...")
            model = PPO.load(LOAD_MODEL_PATH, env=env) 
        else: 
            print("Using newly trained model for visualization.")
            model = PPO.load(SAVE_MODEL_PATH, env=env)


        print("Visualizing learned policy...")
        vis_env = QuadRotorSlalom(render_mode='human') 
        obs, info = vis_env.reset()
        
        terminated = False
        truncated = False
        total_reward_eval = 0
        num_eval_episodes = 5

        for episode in range(num_eval_episodes):
            obs, info = vis_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            print(f"\n--- Evaluation Episode {episode + 1} ---")
            while not terminated and not truncated:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = vis_env.step(action)
                episode_reward += reward
            total_reward_eval += episode_reward
            print(f"Episode {episode + 1} Finished. Reward: {episode_reward:.2f}, Gates Passed: {vis_env.target_gate_index}/{vis_env.num_gates}")
            if vis_env.render_mode == 'human':
                 print("Displaying final trajectory for this episode...")
                 vis_env.render() 

        print(f"\n--- Visualization Finished ---")
        print(f"Average Reward over {num_eval_episodes} episodes: {total_reward_eval / num_eval_episodes:.2f}")
        
        vis_env.close()

    env.close()
    print("Script finished.")