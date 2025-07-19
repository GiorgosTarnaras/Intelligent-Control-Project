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


class QuadRotorFlip(QuadRotor):
    """
    Quadrotor Environment for learning aerial maneuvers including flips.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__() #inherits from 
        self.render_mode = render_mode 
        
        # Task parameters
        self.target_roll = 2 * np.pi  # 360° flip
        self.hover_height = 2.0 #init height 
        self.max_position = 4.0
        # Define State Space
        # Position (3) + Velocity (3) + Euler angles (3) + Angular velocity (3) + Motor forces (4) + Accumulated roll (1) + Goal progress (1)
        self.state_dim = 18
        low_obs = np.array([-np.inf] * self.state_dim, dtype=np.float32)
        high_obs = np.array([np.inf] * self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)
        self.reset_state()
        
        # for visualization 
        self.trajectory = []
        self.euler_history = []
        self.motor_history = []
        self.ang_vel_history = []
        
    def reset_state(self):
        #Initialize all state variables
        self.position = np.array([0.0, 0.0, self.hover_height], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.euler = np.zeros(3, dtype=np.float32)  # [roll, pitch, yaw]
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.motor_forces = np.ones(4, dtype=np.float32) * (self.mass * self.gravity / 4)  # Hover
        self.current_step = 0
        self.total_roll = 0.0
        self.goal_progress = 0.0


    def _get_info(self):
        #Get auxiliary information

        return {
            "roll_error": abs(self.target_roll - self.total_roll),
            "height": self.position[2],
            "roll_degrees": self.total_roll * 180 / np.pi,
            "flip_completed": self.total_roll >= self.target_roll
        }

    def _get_obs(self):
        #Get current observation
        obs = np.concatenate([
            self.position,                    # 3D position
            self.velocity,                    # 3D velocity  
            self.euler,                       # 3D Euler angles
            self.angular_velocity,            # 3D angular velocity
            self.motor_forces / self.max_motor_thrust,  # 4D normalized motor forces
            [self.total_roll / self.target_roll],      # Roll progress [0,1]
            [self.goal_progress]                       # Overall goal progress
        ]).astype(np.float32)
            
        return obs

    def reset(self, seed=None, options=None):
        #Reset environment
        super().reset(seed=seed)
        self.reset_state()
        # Add small random perturbations for robustness
        if seed is not None:
            np.random.seed(seed)

        self.position += np.random.normal(0, 0.1, 3)
        self.position[2] = max(self.position[2], 1.5)  # Ensure above ground
        self.euler += np.random.normal(0, 0.1, 3)
        
        # Reset history
        self.trajectory = [self.position.copy()]
        self.euler_history = [self.euler.copy()]
        self.motor_history = [self.motor_forces.copy()]
        self.ang_vel_history = [self.angular_velocity.copy()]
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        #Step the simulation
        self.current_step += 1
        
        # Clip actions
        motor_thrusts = np.clip(action, self.action_space.low, self.action_space.high)
        self.motor_forces = motor_thrusts
        f_fr, f_rl, f_fl, f_rr = motor_thrusts[0], motor_thrusts[1], motor_thrusts[2], motor_thrusts[3]

        tau_x_body = self.L * (f_rl + f_fl - f_fr - f_rr) / 2.0 
        tau_y_body = self.L * (f_rl + f_rr - f_fr - f_fl) / 2.0 
        tau_z_body = self.c_torque_thrust_ratio * ((motor_thrusts[2] + motor_thrusts[3]) - (motor_thrusts[0] + motor_thrusts[1]))
        total_thrust = np.sum(motor_thrusts)
        
        total_torque = np.array([tau_x_body, tau_y_body, tau_z_body], dtype=np.float32)
        body_force = np.array([0.0,0.0,total_thrust], dtype=np.float32)
        
        # Get rotation matrix
        R = self._get_rotation_matrix(self.euler)
        
        # Apply forces in world frame
        world_force = R @ body_force
        world_force[2] -= self.mass * self.gravity  # Add gravity
                
        # Update linear motion
        acceleration = world_force / self.mass
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        
        
        # Angular acceleration (Euler's equation)
        angular_acceleration = self.inv_I @ (total_torque - np.cross(self.angular_velocity, self.I @ self.angular_velocity))
        self.angular_velocity += angular_acceleration * self.dt
        
        # Update Euler angles
        euler_rates = self._body_rates_to_euler_rates(self.euler, self.angular_velocity)
        prev_roll = self.euler[0]
        self.euler += euler_rates * self.dt
        
        # Keep angles in reasonable range
        self.euler[0] = self.euler[0] % (2 * np.pi)  # Roll
        self.euler[1] = np.clip(self.euler[1], -np.pi/2, np.pi/2)  # Pitch
        self.euler[2] = self.euler[2] % (2 * np.pi)  # Yaw
        
        # Track total roll
        roll_change = self.euler[0] - prev_roll
        if roll_change > np.pi:
            roll_change -= 2 * np.pi
        elif roll_change < -np.pi:
            roll_change += 2 * np.pi
        self.total_roll += abs(roll_change)
        
        # Update goal progress
        self.goal_progress = min(self.total_roll / self.target_roll, 1.0)
        
        # Store history
        self.trajectory.append(self.position.copy())
        self.euler_history.append(self.euler.copy())
        self.motor_history.append(self.motor_forces.copy())
        self.ang_vel_history.append(self.angular_velocity.copy())
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _calculate_reward(self):
        reward = 0.0
        
        # Progress reward (main objective)
        progress_reward = (self.goal_progress**0.5) * 100.0 # make it non linear 
        reward += progress_reward
        
        # Completion bonus
        if self.total_roll >= self.target_roll:
            reward += 500.0
        
        # Position stability (hover around target position)
        target_pos = np.array([0, 0, self.hover_height])
        pos_error = np.linalg.norm(self.position - target_pos)
        reward -= pos_error * 100.0 # high penalty for losing starting pos
        
        # Velocity penalty (encourage smooth motion)
        vel_penalty = np.linalg.norm(self.velocity) * 0.5
        reward -= vel_penalty
        
        # Angular velocity penalty for non-roll axes
        unwanted_angular_vel = np.linalg.norm(self.angular_velocity[1:])  # pitch and yaw rates
        reward -= unwanted_angular_vel * 2.0
        
        # Motor efficiency (penalize extreme motor commands)
        motor_variance = np.var(self.motor_forces)
        reward -= motor_variance * 0.1
        
        # Orientation stability (penalize excessive pitch/yaw)
        orientation_penalty = abs(self.euler[1]) + abs(self.euler[2])  # pitch + yaw
        reward -= orientation_penalty * 10.0
        
        # Altitude maintenance bonus
        altitude_error = abs(self.position[2] - self.hover_height)
        if altitude_error < 0.25:
            reward += 5.0
        
        return reward

    def _check_termination(self):
        #Check if episode should terminate
        terminated = False
        truncated = False
        
        # Crash detection
        if self.position[2] < self.collision_radius:
            terminated = True
        
        # Out of bounds
        if np.any(np.abs(self.position[:2]) > self.max_position) or self.position[2] > 2 * self.max_position:
            terminated = True
        
        # Excessive velocities
        if np.linalg.norm(self.velocity) > self.max_velocity:
            terminated = True
        
        # Excessive angular velocities
        if np.linalg.norm(self.angular_velocity) > self.max_angular_velocity:
            terminated = True
        
        # Success condition
        if self.total_roll >= self.target_roll:
            truncated = True
        
        # Time limit
        if self.current_step >= self.max_steps_per_episode:
            truncated = True
        
        return terminated, truncated

    def render(self):
        #Render the environment
        if self.render_mode in ['human', 'rgb_array']:
            fig = plt.figure(figsize=(15, 10))
            
            # 3D trajectory
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            traj = np.array(self.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.7, linewidth=2)
            ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, label='Start')
            ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, label='End')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory')
            ax1.legend()
            
            # Euler angles
            ax2 = fig.add_subplot(2, 3, 2)
            times = np.arange(len(self.euler_history)) * self.dt
            euler_deg = np.array(self.euler_history) * 180 / np.pi
            ax2.plot(times, euler_deg[:, 0], label='Roll', linewidth=2)
            ax2.plot(times, euler_deg[:, 1], label='Pitch', alpha=0.7)
            ax2.plot(times, euler_deg[:, 2], label='Yaw', alpha=0.7)
            ax2.axhline(y=360, color='r', linestyle='--', label='Target (360°)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Angle (degrees)')
            ax2.set_title('Euler Angles')
            ax2.legend()
            ax2.grid(True)
            
            # Motor commands
            ax3 = fig.add_subplot(2, 3, 3)
            motor_history = np.array(self.motor_history)
            for i in range(4):
                ax3.plot(times, motor_history[:, i], label=f'Motor {i+1}')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Force (N)')
            ax3.set_title('Motor Forces')
            ax3.legend()
            ax3.grid(True)
            
            # Position vs time
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.plot(times, traj[:, 0], label='X')
            ax4.plot(times, traj[:, 1], label='Y') 
            ax4.plot(times, traj[:, 2], label='Z', linewidth=2)
            ax4.axhline(y=self.hover_height, color='r', linestyle='--', alpha=0.5, label='Target Height')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Position (m)')
            ax4.set_title('Position vs Time')
            ax4.legend()
            ax4.grid(True)
            
            # Angular velocities
            ax5 = fig.add_subplot(2, 3, 5)
            ang_vel_history = np.array(self.ang_vel_history) * 180 / np.pi
            ax5.plot(times, ang_vel_history[:, 0], label='Omega X')
            ax5.plot(times, ang_vel_history[:, 1], label='Omega Y')
            ax5.plot(times, ang_vel_history[:, 2], label='Omega Z')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Angular Velocity (deg/s)')
            ax5.set_title('Angular Velocities')
            ax5.legend()
            ax5.grid(True)
            
            # Performance metrics
            ax6 = fig.add_subplot(2, 3, 6)
            metrics_text = f"""Performance Metrics:
                                    
                        Total Roll: {self.total_roll * 180 / np.pi:.1f}°
                        Target: 360°
                        Progress: {self.goal_progress * 100:.1f}%

                        Final Position:
                        X: {self.position[0]:.2f} m
                        Y: {self.position[1]:.2f} m  
                        Z: {self.position[2]:.2f} m

                        Duration: {self.current_step * self.dt:.2f} s
                        Steps: {self.current_step}

                        Flip Completed: {self.total_roll >= self.target_roll}
                            """


            ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=12)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis('off')
            ax6.set_title('Performance Summary')
            
            plt.tight_layout()
            
            if self.render_mode == 'human':
                plt.show()
            else:
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img

    def close(self):
        plt.close('all')


# train our model for 10million steps
if __name__ == "__main__":
    TRAIN_MODEL = False #make true to train the model
    LOAD_MODEL_PATH = "quad_flip_ppo.zip"
    SAVE_MODEL_PATH = "quad_flip_ppo.zip"
    TOTAL_TIMESTEPS = 10000000  # complex task requires a lot of timesteps
    log_dir = "./quad_flip_tensorboard/"
    # Environment setup
    print("Creating realistic quadrotor environment...")
    env = DummyVecEnv([lambda: QuadRotorFlip(render_mode='rgb_array')])
    
    if TRAIN_MODEL:
        print(f"Training model for {TOTAL_TIMESTEPS} timesteps...")
        # More sophisticated PPO parameters for complex task
        model = PPO.load("quad_flip_ppo.zip", env=env)
        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        print(f"Saving model to {SAVE_MODEL_PATH}")
        model.save(SAVE_MODEL_PATH)
    
    # Load and test
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading model from {LOAD_MODEL_PATH}...")
        model = PPO.load(LOAD_MODEL_PATH, env=env)
        
        # Test the trained model
        print("Testing trained model...")
        test_env = QuadRotorFlip(render_mode='human')
        
        for episode in range(3):  # Test multiple episodes
            print(f"\n--- Episode {episode + 1} ---")
            obs, info = test_env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            
            while not terminated and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
            
            print(f"Episode {episode + 1} Results:")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Roll Achieved: {info['roll_degrees']:.1f}°")
            print(f"Flip Completed: {info['flip_completed']}")
            print(f"Final Height: {info['height']:.2f}m")
            print(f"Steps: {test_env.current_step}")
            
            if episode == 0:  # Show detailed visualization for first episode
                test_env.render()
        
        test_env.close()
    else:
        print(f"Model file {LOAD_MODEL_PATH} not found. Please train first.")
    
    env.close()