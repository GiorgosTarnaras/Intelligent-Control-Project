import numpy as np
import pybullet as p
import pybullet_data
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from quadrotor_flip import QuadRotorFlip 

class QuadrotorVisualizer:
    
    def __init__(self):
        self.height_target = 2.0
        self.gravity = 9.81
        self.mass = 0.5
        self.dt = 0.05
        
        # Setup PyBullet
        self.client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.gravity, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client_id)
        
        # Create realistic quadrotor visual
        self.quad_id = self._create_quadrotor_body([0, 0, self.height_target])
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)
    
    def _create_quadrotor_body(self, position):
        
        # Create main body
        body_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.1, 0.1, 0.03], 
            rgbaColor=[0.2, 0.2, 0.2, 1.0]  # Dark gray body
        )
        body_collision = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[0.1, 0.1, 0.03]
        )
        
        # Create the main body
        quad_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=position,
            physicsClientId=self.client_id
        )
        
        # Create arms and rotors as separate visual-only bodies
        arm_length = 0.15/np.sqrt(2)
        rotor_positions = [
            [arm_length, 0, 0],      # Front
            [-arm_length, 0, 0],     # Back
            [0, arm_length, 0],      # Right
            [0, -arm_length, 0]      # Left
        ]
        
        rotor_colors = [
            [1, 0.2, 0.2, 1],    # Red (front)
            [0.2, 1, 0.2, 1],    # Green (back)  
            [0.2, 0.2, 1, 1],    # Blue (right)
            [1, 1, 0.2, 1]       # Yellow (left)
        ]
        
        # Store arm and rotor IDs for updating
        self.arm_ids = []
        self.rotor_ids = []
        
        for i, (rel_pos, color) in enumerate(zip(rotor_positions, rotor_colors)):
            # Create arm
            if i < 2:  # Front-back arms (X-axis)
                arm_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[arm_length * 0.8, 0.015, 0.01],
                    rgbaColor=[0.4, 0.4, 0.4, 1.0]
                )
            else:  # Left-right arms (Y-axis)
                arm_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[0.015, arm_length * 0.8, 0.01],
                    rgbaColor=[0.4, 0.4, 0.4, 1.0]
                )
            
            arm_id = p.createMultiBody(
                baseMass=0,  # Visual only
                baseVisualShapeIndex=arm_visual,
                basePosition=[position[0] + rel_pos[0] * 0.5, 
                             position[1] + rel_pos[1] * 0.5, 
                             position[2] + 0.02],
                physicsClientId=self.client_id
            )
            self.arm_ids.append(arm_id)
            
            # Create rotor
            rotor_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.08,
                length=0.02,
                rgbaColor=color
            )
            
            rotor_id = p.createMultiBody(
                baseMass=0,  # Visual only
                baseVisualShapeIndex=rotor_visual,
                basePosition=[position[0] + rel_pos[0], 
                             position[1] + rel_pos[1], 
                             position[2] + 0.05],
                physicsClientId=self.client_id
            )
            self.rotor_ids.append(rotor_id)
        
        return quad_id
    
    def update_quad_pose(self, position, roll):
        # Update main body
        p.resetBasePositionAndOrientation(
            self.quad_id,
            position,
            p.getQuaternionFromEuler([roll, 0, 0]),
            physicsClientId=self.client_id
        )
        
        # Update arms and rotors to follow the main body
        arm_length = 0.15/np.sqrt(2)
        rotor_positions = [
            [arm_length, 0, 0],      # Front
            [-arm_length, 0, 0],     # Back
            [0, arm_length, 0],      # Right
            [0, -arm_length, 0]      # Left
        ]
        
        # Rotate the relative positions based on roll
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        
        for i, rel_pos in enumerate(rotor_positions):
            # Rotate position around X-axis (roll)
            rotated_y = rel_pos[1] * cos_roll - rel_pos[2] * sin_roll
            rotated_z = rel_pos[1] * sin_roll + rel_pos[2] * cos_roll
            
            # Update arm position
            arm_pos = [
                position[0] + rel_pos[0] * 0.5,
                position[1] + rotated_y * 0.5,
                position[2] + rotated_z * 0.5 + 0.02
            ]
            p.resetBasePositionAndOrientation(
                self.arm_ids[i],
                arm_pos,
                p.getQuaternionFromEuler([roll, 0, 0]),
                physicsClientId=self.client_id
            )
            
            # Update rotor position
            rotor_pos = [
                position[0] + rel_pos[0],
                position[1] + rotated_y,
                position[2] + rotated_z + 0.05
            ]
            p.resetBasePositionAndOrientation(
                self.rotor_ids[i],
                rotor_pos,
                p.getQuaternionFromEuler([roll, 0, 0]),
                physicsClientId=self.client_id
            )
    
    def step_simulation(self):
        p.stepSimulation(physicsClientId=self.client_id)
        time.sleep(3*self.dt)  # Slow down for better visualization
    
    def close(self):
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None

def run_visualization(model_path="quad_flip_ppo.zip", max_episodes=3):
    """Run the trained model with PyBullet visualization."""
    
    # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    
    # Load your existing environment and trained model
    dummy_env = QuadRotorFlip(render_mode=None)  # Use your module
    model = PPO.load(model_path, env=DummyVecEnv([lambda: dummy_env]))
    
    # Create physics environment and visualizer
    env = QuadRotorFlip(render_mode=None)  # Physics only
    visualizer = QuadrotorVisualizer()
    video_path = "simulation.mp4"
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
    print("Running simulation with PyBullet GUI...")
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0
        max_roll_rate = 0.0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not terminated and not truncated:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the physics environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update visualization
            visualizer.update_quad_pose(env.position, env.euler[0])
            visualizer.step_simulation()
            
            # Track metrics
            total_reward += reward
            steps += 1
            roll_degrees = env.total_roll * 180 / np.pi
            roll_rate = abs(env.angular_velocity[0]) * 180 / np.pi
            max_roll_rate = max(max_roll_rate, roll_rate)
            flip_completed = env.total_roll >= 2 * np.pi
            
            # Print status every 5 steps
            if steps % 5 == 0:
                print(f"Step: {steps}, Alt: {env.position[2]:.2f}m, "
                      f"Roll: {roll_degrees:.1f}°, Rate: {roll_rate:.1f}°/s, "
                      f"Flip: {flip_completed}")
        
        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
        print(f"Final Alt: {env.position[2]:.2f}m, Roll: {roll_degrees:.1f}°")
        print(f"Max Roll Rate: {max_roll_rate:.1f}°/s")
        print(f"Flip Completed: {flip_completed}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        #time.sleep(10)
    

    p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
    visualizer.close()

    print("\nVisualization finished.")

if __name__ == "__main__":

    run_visualization("quad_flip_ppo.zip", max_episodes=3)