import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
from quadrotor import QuadRotor

# Quadrotor Dive Environment
class QuadRotorDive(QuadRotor):
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        #Env Params for Dive , In SI units
        self.initial_altitude_mean = 20.0 
        self.initial_altitude_std = 3.0   
        self.initial_xy_spread = 2.0      
        self.dive_target_z = 1.0          
        self.dive_target_xy = np.array([0.0, 0.0], dtype=np.float32) # Target XY coordinate on the Z-plane
        self.success_z_threshold = 0.5    
        self.success_xy_threshold = 1.5   
        self.success_vz_threshold = 1.0   

        self.max_steps_per_episode = 600 
        self.max_tilt_angle_rad = np.pi / 3 # beyond this penalize heavily or terminate

        # Observation Space 
        # [pos(3), vel(3), quat(4), ang_vel(3), relative_pos_to_dive_target(3)] = 16
        low_obs = np.array([-np.inf] * 16, dtype=np.float32)
        high_obs = np.array([np.inf] * 16, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)
        
        
        # Action Space 
        # Action: Thrust for each of the 4 motors [F1, F2, F3, F4]
        # F1: front-right, F2: front-left, F3: rear-left, F4: rear-right 
        #   1(FR,CW)    0(FL,CCW)
        #      \      /
        #       \    /
        #        ----    --> Nose (positive X body)
        #       /    \
        #      /      \
        #   2(RL,CW)    3(RR,CCW)
        # Torque calculation:
        # Roll: action[0], action[2] vs action[1], action[3]
        # Pitch: action[0], action[1] vs action[2], action[3]
        # Yaw: action[0], action[3] (CCW) vs action[1], action[2] (CW)
        
        
        hover_thrust_per_motor = (self.mass * self.gravity) / 4.0
        self.max_thrust_per_motor = hover_thrust_per_motor * 3.0
        min_thrust_per_motor = 0.0
        low_action = np.array([min_thrust_per_motor] * 4, dtype=np.float32)
        high_action = np.array([self.max_thrust_per_motor] * 4, dtype=np.float32)
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)

        # State Vars
        self.quad_pos = np.zeros(3, dtype=np.float32)
        self.quad_vel = np.zeros(3, dtype=np.float32)
        self.quad_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # w,x,y,z
        self.quad_ang_vel = np.zeros(3, dtype=np.float32) # wx, wy, wz in body frame
        self.start_altitude = self.initial_altitude_mean #keeping track of initial Z

        self.current_step = 0
        self.trajectory = []

    def _get_obs(self):
        rel_pos_to_dive_target = np.concatenate([
            self.dive_target_xy - self.quad_pos[:2],
            [self.dive_target_z - self.quad_pos[2]]
        ]).astype(np.float32)

        return np.concatenate([
            self.quad_pos,
            self.quad_vel,
            self.quad_quat,
            self.quad_ang_vel,
            rel_pos_to_dive_target
        ]).astype(np.float32)

    def _get_info(self):
        dist_to_target_z = abs(self.quad_pos[2] - self.dive_target_z)
        dist_to_target_xy = np.linalg.norm(self.quad_pos[:2] - self.dive_target_xy)
        return {
            "current_pos": self.quad_pos.copy(),
            "current_vel": self.quad_vel.copy(),
            "dist_to_target_z": dist_to_target_z,
            "dist_to_target_xy": dist_to_target_xy,
            "altitude": self.quad_pos[2]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # High attitude initiation and slight XY randomization
        self.start_altitude = self.np_random.normal(self.initial_altitude_mean, self.initial_altitude_std)
        start_x = self.np_random.uniform(-self.initial_xy_spread, self.initial_xy_spread)
        start_y = self.np_random.uniform(-self.initial_xy_spread, self.initial_xy_spread)
        self.quad_pos = np.array([start_x, start_y, self.start_altitude], dtype=np.float32)

        self.quad_vel = np.zeros(3, dtype=np.float32) # Zero velocity initiallazation (convenient but not very realistic)
        self.quad_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Start upright
        self.quad_ang_vel = np.zeros(3, dtype=np.float32)

       
        self.current_step = 0
        self.trajectory = [self.quad_pos.copy()]
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        pos_before_step = self.quad_pos.copy()
        vel_before_step = self.quad_vel.copy()

        # Apply Physics 
        motor_thrusts = np.clip(action, self.action_space.low, self.action_space.high)
        f_fr, f_rl, f_fl, f_rr = motor_thrusts[0], motor_thrusts[1], motor_thrusts[2], motor_thrusts[3]
        total_thrust = np.sum(motor_thrusts)

        # X-config motor numbering & effect:
        # Action order: [F_FR, F_RL, F_FL, F_RR] (FrontRight, RearLeft, FrontLeft, RearRight)
        # Torques in body frame
        tau_x_body = self.L * (f_rl + f_fl - f_fr - f_rr) / 2.0 # Roll
        tau_y_body = self.L * (f_rl + f_rr - f_fr - f_fl) / 2.0 # Pitch
        tau_z_body = self.c_torque_thrust_ratio * ((f_fl + f_rr) - (f_fr + f_rl)) # Yaw

        torques_body = np.array([tau_x_body, tau_y_body, tau_z_body], dtype=np.float32)
        ang_vel_dot_body = self.inv_I @ (torques_body - np.cross(self.quad_ang_vel, self.I @ self.quad_ang_vel))
        self.quad_ang_vel += ang_vel_dot_body * self.dt

        q_dot = 0.5 * self.quaternion_multiply(np.concatenate(([0.0], self.quad_ang_vel)), self.quad_quat)
        self.quad_quat += q_dot * self.dt
        self.quad_quat /= np.linalg.norm(self.quad_quat) # Normalize

        R_body_to_world = self.quaternion_to_rotation_matrix(self.quad_quat)
        thrust_vector_body = np.array([0, 0, total_thrust], dtype=np.float32)
        acceleration_world = (R_body_to_world @ thrust_vector_body) / self.mass \
                             + np.array([0, 0, -self.gravity], dtype=np.float32)
        self.quad_vel += acceleration_world * self.dt
        self.quad_pos += self.quad_vel * self.dt
        self.trajectory.append(self.quad_pos.copy())

        # Reward calc for Dive 
        terminated = False
        truncated = False
        reward = 0.0

        #Progress towards target Z
        prev_dist_to_target_z = abs(pos_before_step[2] - self.dive_target_z)
        current_dist_to_target_z = abs(self.quad_pos[2] - self.dive_target_z)
        reward += (prev_dist_to_target_z - current_dist_to_target_z) * 1.5 # Stronger weight for Z progress

        #Drift penalty from target XY
        current_dist_to_target_xy = np.linalg.norm(self.quad_pos[:2] - self.dive_target_xy)
        reward -= current_dist_to_target_xy * 0.15

        #Stability penalties
        reward -= np.linalg.norm(self.quad_ang_vel) * 0.01 # Ang vel penalty
        up_vector_world = R_body_to_world[:, 2] 
        cos_tilt = up_vector_world[2] 
        if cos_tilt < np.cos(self.max_tilt_angle_rad): 
            reward -= (np.cos(self.max_tilt_angle_rad) - cos_tilt) * 10.0 # Penalize significant tilt
        if cos_tilt < 0.0: 
            reward -= 150.0
            terminated = True

        # Time penalty
        reward -= 0.1 # encourage faster dives

        
        # Ground collision
        if self.quad_pos[2] < self.collision_radius:
            reward -= 150.0
            terminated = True

        # Successful completion
        if not terminated and current_dist_to_target_z < self.success_z_threshold \
           and current_dist_to_target_xy < self.success_xy_threshold \
           and abs(self.quad_vel[2]) < self.success_vz_threshold :
            reward += 250.0 
            truncated = True # Successfully completed task

        # Flew too high 
        if self.quad_pos[2] > self.start_altitude + 10.0: #10m above initial start
            reward -= 100.0
            terminated = True
            
        # OOB (out of bounds
        max_xy_extent = 2 * self.initial_xy_spread + 20.0
        max_z_extent = self.initial_altitude_mean + self.initial_altitude_std + 15.0
        if np.any(np.abs(self.quad_pos[:2]) > max_xy_extent) or \
           self.quad_pos[2] > max_z_extent or self.quad_pos[2] < -5:
            reward -= 100.0
            if not truncated : terminated = True 


        if self.current_step >= self.max_steps_per_episode:
            if not truncated: # If truncated by steps but not by success
                reward -= 50.0 
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array' or self.render_mode == 'human':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='.', markersize=2, linestyle='-', label='Quadrotor Path')
            if len(traj) > 0:
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, label='Start', depthshade=True)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, label='End', depthshade=True)

            # Plot target dive plane (simplified)
            target_patch_size = self.success_xy_threshold
            xx, yy = np.meshgrid(
                [self.dive_target_xy[0] - target_patch_size, self.dive_target_xy[0] + target_patch_size],
                [self.dive_target_xy[1] - target_patch_size, self.dive_target_xy[1] + target_patch_size]
            )
            zz = np.full_like(xx, self.dive_target_z)
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan', label='Target Dive Zone')

            # Plot ground plane
            ground_extent = max(self.initial_xy_spread * 2, 10)
            xx_ground, yy_ground = np.meshgrid([-ground_extent, ground_extent], [-ground_extent, ground_extent])
            zz_ground = np.zeros_like(xx_ground)
            ax.plot_surface(xx_ground, yy_ground, zz_ground, alpha=0.2, color='brown', label='Ground (Z=0)')


            max_render_x = max(np.abs(traj[:,0])) if len(traj)>0 else self.initial_xy_spread + 5
            max_render_y = max(np.abs(traj[:,1])) if len(traj)>0 else self.initial_xy_spread + 5
            max_render_z = self.initial_altitude_mean + self.initial_altitude_std + 5

            ax.set_xlim(-max_render_x, max_render_x)
            ax.set_ylim(-max_render_y, max_render_y)
            ax.set_zlim(0, max_render_z)
            ax.set_xlabel("X position (m)")
            ax.set_ylabel("Y position (m)")
            ax.set_zlabel("Z position (Height) (m)")
            ax.set_title(f"Quadrotor Dive Trajectory (Step: {self.current_step})")
            ax.legend()
            plt.grid(True)

            if self.render_mode == 'human':
                plt.show(block=False)
                plt.pause(0.01) # small pause to update the plot window
            elif self.render_mode == 'rgb_array':
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img
        return None

    def close(self):
        plt.close('all')


#we train this for 4million steps 
if __name__ == "__main__":
    TRAIN_MODEL = False #make True to train from start
    LOAD_MODEL_PATH = "quad_dive_ppo.zip"
    SAVE_MODEL_PATH = "quad_dive_ppo.zip"
    TOTAL_TIMESTEPS = 4_000_000 

    log_dir = "./quad_physics_dive_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    print("Creating Dive environment with detailed physics...")
    env = DummyVecEnv([lambda: QuadRotorDive(render_mode='rgb_array')])


    if TRAIN_MODEL:
        print(f"Training PPO model for Dive task for {TOTAL_TIMESTEPS} timesteps...")
        model = PPO("MlpPolicy",# in comments are the initial params (in case we have changed them with worst result)
                    env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=3e-4, # 3e-4
                    n_steps=2048,       # 2048
                    batch_size=64,      # 64
                    gamma=0.99,         # 0.99
                    gae_lambda=0.95,    # 0.95
                    ent_coef=0.005,     # 0.05
                    # policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Example custom network
                   )
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        print(f"Saving model to {SAVE_MODEL_PATH}")
        model.save(SAVE_MODEL_PATH)
        del model # deleting trained model if  reload

    if not os.path.exists(LOAD_MODEL_PATH) and not TRAIN_MODEL:
        print(f"Error: Model file not found at {LOAD_MODEL_PATH}. Train the model first or set TRAIN_MODEL=True.")
    else:
        if os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading model from {LOAD_MODEL_PATH}...")
            model = PPO.load(LOAD_MODEL_PATH, env=env)
        else: 
            print("Using newly trained model for visualization.")
            model = PPO.load(SAVE_MODEL_PATH, env=env) 

        print("Visualizing learned policy for Dive task...")
        vis_env = QuadRotorDive(render_mode='human')
        obs, info = vis_env.reset()

        total_reward_eval = 0
        num_eval_episodes = 5
        successful_dives = 0

        for episode in range(num_eval_episodes):
            obs, info = vis_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            print(f"\n--- Evaluation Episode {episode + 1} ---")
            initial_pos = obs[:3].copy() 
            print(f"Starting at: X={initial_pos[0]:.2f}, Y={initial_pos[1]:.2f}, Z={initial_pos[2]:.2f}")

            for step_num in range(vis_env.max_steps_per_episode):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = vis_env.step(action)
                episode_reward += reward
                if vis_env.render_mode == 'human':
                     vis_env.render() 

                if terminated or truncated:
                    break
            
            total_reward_eval += episode_reward
            final_pos = info.get("current_pos", obs[:3])
            final_alt = info.get("altitude", obs[2])
            dist_z = info.get("dist_to_target_z", float('inf'))
            dist_xy = info.get("dist_to_target_xy", float('inf'))
            final_vz = info.get("current_vel",[0,0,0])[2] if "current_vel" in info else obs[5]


            print(f"Episode {episode + 1} Finished. Steps: {step_num+1}")
            print(f"  Final Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}, Z={final_alt:.2f}")
            print(f"  Vz: {final_vz:.2f} m/s")
            print(f"  Distance to Target Z: {dist_z:.2f} m, XY: {dist_xy:.2f} m")
            print(f"  Episode Reward: {episode_reward:.2f}")

            if dist_z < vis_env.success_z_threshold and \
               dist_xy < vis_env.success_xy_threshold and \
               abs(final_vz) < vis_env.success_vz_threshold and \
               not terminated : # checking it wasn't terminated for a crash
                successful_dives +=1
                print("  Dive Considered SUCCESSFUL!")
            elif terminated:
                print("  Dive Terminated (e.g., crash, flip).")
            else:
                print("  Dive Incomplete or Missed Target.")

            if vis_env.render_mode == 'human' and (terminated or truncated):
                print("Displaying final trajectory for this episode (human mode)...")
                # vis_env.render() 
                plt.show(block=True) 
            elif vis_env.render_mode != 'human' and (terminated or truncated): 
                _ = vis_env.render()


        print(f"\n--- Dive Visualization Finished ---")
        print(f"Average Reward over {num_eval_episodes} episodes: {total_reward_eval / num_eval_episodes:.2f}")
        print(f"Successful dives: {successful_dives}/{num_eval_episodes}")

        vis_env.close()

    env.close()
    print("Dive script finished.")