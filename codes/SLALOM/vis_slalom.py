import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from quadrotor_slalom import QuadRotorSlalom

class QuadRotorSlalomVis(QuadRotorSlalom):
    # Updated Env for visualization
    
    def __init__(self, render_mode='human', num_gates=5, course_width=10, gate_spacing=15, gate_size=2.0):
        super().__init__(render_mode)
        self.gate_thickness = 0.2
        self.update_visual_every_n_steps = 2  
        self.trail_update_interval = 5 
        self.camera_update_interval = 10  
        self.max_trail_points = 50 
        
        self.physics_client = None
        self.quad_id = None
        self.gate_ids = []
        self.trail_points = []
        self.trail_line_ids = []
        self.step_counter = 0
        

        self.last_camera_pos = np.zeros(3)
        self.rotor_positions_body = np.array([
            [self.L/np.sqrt(2), self.L/np.sqrt(2), 0.1],
            [self.L/np.sqrt(2), -self.L/np.sqrt(2), 0.1],
            [-self.L/np.sqrt(2), self.L/np.sqrt(2), 0.1],
            [-self.L/np.sqrt(2), -self.L/np.sqrt(2), 0.1]
        ], dtype=np.float32)
        

        self._setup_pybullet()
    
    def _setup_pybullet(self):
        #PyBullet physics simulation and visual elements.
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        

        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)  # Disable GUI panels
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        

        p.setGravity(0, 0, -self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.dt, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(numSolverIterations=5, physicsClientId=self.physics_client)  # Reduced from default 50
        

        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        

        self._create_simplified_quadrotor()
        

        self._create_simplified_gates()
        

        self._set_camera()
    
    def _create_simplified_quadrotor(self):
        """Create a more realistic visual quadrotor representation while keeping it optimized."""

        body_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05], physicsClientId=self.physics_client)
        body_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.1, 0.1, 0.035], 
            rgbaColor=[0.3, 0.3, 0.3, 1.0],  
            physicsClientId=self.physics_client
        )
        
        arm_length = 0.15
        arm_half_extents = [0.075, 0.015, 0.015]  
        arm_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=arm_half_extents, 
            rgbaColor=[0.2, 0.2, 0.2, 1.0],  
            physicsClientId=self.physics_client
        )
        arm_collision = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=arm_half_extents, 
            physicsClientId=self.physics_client
        )

        prop_radius = 0.08
        prop_length = 0.01  
        prop_visual = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=prop_radius, 
            length=prop_length,  
            rgbaColor=[0.1, 0.1, 0.1, 0.8],  
            physicsClientId=self.physics_client
        )
        
      
        link_masses = [0.01] * 4  
        link_collision_shapes = [arm_collision] * 4
        link_visual_shapes = [arm_visual] * 4
        link_positions = [
            [arm_length, 0, 0], 
            [-arm_length, 0, 0], 
            [0, arm_length, 0],  
            [0, -arm_length, 0]  
        ]
        link_orientations = [[0, 0, 0, 1]] * 4  
        link_inertial_frame_positions = [[0, 0, 0]] * 4
        link_inertial_frame_orientations = [[0, 0, 0, 1]] * 4
        link_parent_indices = [0] * 4 
        link_joint_types = [p.JOINT_FIXED] * 4
        link_joint_axis = [[0, 0, 0]] * 4


        self.quad_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=body_shape,
            baseVisualShapeIndex=body_visual,
            basePosition=[0, 0, 1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frame_positions,
            linkInertialFrameOrientations=link_inertial_frame_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            physicsClientId=self.physics_client
        )


        self.propeller_visual_ids = []
        prop_positions = [
            [arm_length + 0.05, 0, 0.05],   
            [-arm_length - 0.05, 0, 0.05],  
            [0, arm_length + 0.05, 0.05],   
            [0, -arm_length - 0.05, 0.05]  
        ]
        for pos in prop_positions:
            prop_id = p.createMultiBody(
                baseMass=0.0,  
                baseVisualShapeIndex=prop_visual,
                basePosition=pos,  
                physicsClientId=self.physics_client
            )
            self.propeller_visual_ids.append(prop_id)

        self.propeller_line_ids = []
    
    def _create_simplified_gates(self):
        #simplified visual gate structures
        self.gate_ids = []
        
        for i, gate_info in enumerate(self.gates):
            pos = gate_info['position']
            
           
            half_size = self.gate_size / 2
            
  
            corners = [
                [pos[0] - half_size, pos[1] - half_size, pos[2] - half_size],
                [pos[0] + half_size, pos[1] - half_size, pos[2] - half_size],
                [pos[0] + half_size, pos[1] + half_size, pos[2] - half_size],
                [pos[0] - half_size, pos[1] + half_size, pos[2] - half_size],
                [pos[0] - half_size, pos[1] - half_size, pos[2] + half_size],
                [pos[0] + half_size, pos[1] - half_size, pos[2] + half_size],
                [pos[0] + half_size, pos[1] + half_size, pos[2] + half_size],
                [pos[0] - half_size, pos[1] + half_size, pos[2] + half_size]
            ]
            

            gate_lines = []

            for j in range(4):
                line_id = p.addUserDebugLine(corners[j], corners[(j+1)%4], 
                                           lineColorRGB=[0.8, 0.2, 0.2], lineWidth=3,
                                           physicsClientId=self.physics_client)
                gate_lines.append(line_id)
            

            for j in range(4, 8):
                line_id = p.addUserDebugLine(corners[j], corners[4 + (j+1-4)%4], 
                                           lineColorRGB=[0.8, 0.2, 0.2], lineWidth=3,
                                           physicsClientId=self.physics_client)
                gate_lines.append(line_id)
            

            for j in range(4):
                line_id = p.addUserDebugLine(corners[j], corners[j+4], 
                                           lineColorRGB=[0.8, 0.2, 0.2], lineWidth=3,
                                           physicsClientId=self.physics_client)
                gate_lines.append(line_id)
            
            self.gate_ids.append(gate_lines)
    
    def _set_camera(self):
        #Set up the camera to follow the drone
        p.resetDebugVisualizerCamera(cameraDistance=12,
                                   cameraYaw=45,
                                   cameraPitch=-30,
                                   cameraTargetPosition=[0, 0, 10],
                                   physicsClientId=self.physics_client)
    
    def _update_visual_elements(self):
        #only update when necessary.
        self.step_counter += 1
        

        pybullet_quat = [self.quad_quat[1], self.quad_quat[2], self.quad_quat[3], self.quad_quat[0]]
        p.resetBasePositionAndOrientation(self.quad_id, 
                                        self.quad_pos, 
                                        pybullet_quat,
                                        physicsClientId=self.physics_client)
        

        if self.step_counter % self.update_visual_every_n_steps == 0:
            self._update_propeller_indicators()
            
        if self.step_counter % self.trail_update_interval == 0:
            self._update_trail()
            
        if self.step_counter % self.camera_update_interval == 0:
            self._update_camera()
            
        if self.step_counter % (self.update_visual_every_n_steps * 2) == 0:
            self._update_gate_colors()
    
    def _update_propeller_indicators(self):
        #simulate spinning effect.

        for line_id in self.propeller_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.physics_client)
            except:
                pass
        self.propeller_line_ids.clear()

        R = self.quaternion_to_rotation_matrix(self.quad_quat)
        prop_positions = [
            [self.L + 0.05, 0, 0.05],   
            [-self.L - 0.05, 0, 0.05],  
            [0, self.L + 0.05, 0.05],  
            [0, -self.L - 0.05, 0.05]   
        ]
        for i, (prop_id, pos_body) in enumerate(zip(self.propeller_visual_ids, prop_positions)):
            pos_world = self.quad_pos + R @ np.array(pos_body, dtype=np.float32)
           
            angle = (self.step_counter * 0.5) % (2 * math.pi)  
            prop_quat = [0, 0, math.sin(angle/2), math.cos(angle/2)]  
            p.resetBasePositionAndOrientation(
                prop_id,
                pos_world,
                prop_quat,
                physicsClientId=self.physics_client
            )
    
    def _update_trail(self):
        #Optimized trail update.
        self.trail_points.append(self.quad_pos.copy())
        
      
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points.pop(0)
     
        for line_id in self.trail_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.physics_client)
            except:
                pass
        self.trail_line_ids.clear()
        
       
        if len(self.trail_points) > 1:
            step = max(1, len(self.trail_points) // 20)  
            for i in range(0, len(self.trail_points) - step, step):
                line_id = p.addUserDebugLine(self.trail_points[i], 
                                           self.trail_points[i + step],
                                           lineColorRGB=[0, 1, 0],
                                           lineWidth=2,
                                           physicsClientId=self.physics_client)
                self.trail_line_ids.append(line_id)
    
    def _update_camera(self):
        pos_diff = np.linalg.norm(self.quad_pos - self.last_camera_pos)
        if pos_diff > 2.0:  
            p.resetDebugVisualizerCamera(cameraDistance=12,
                                       cameraYaw=45,
                                       cameraPitch=-30,
                                       cameraTargetPosition=self.quad_pos,
                                       physicsClientId=self.physics_client)
            self.last_camera_pos = self.quad_pos.copy()
    
    def _update_gate_colors(self):
        for i, (gate_info, gate_lines) in enumerate(zip(self.gates, self.gate_ids)):
            if gate_info['passed']:
                color = [0.2, 0.8, 0.2]  
            elif i == self.target_gate_index:
                color = [0.8, 0.8, 0.2]  
            else:
                color = [0.8, 0.2, 0.2] 
           
            pass
    
    def _get_obs(self):
        #same format as training environment
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
    
    def reset(self, seed=None, options=None):
        #env reset
        super().reset(seed=seed)        
        
        self.step_counter = 0       
        self.trail_points.clear()
        for line_id in self.trail_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.physics_client)
            except:
                pass
        self.trail_line_ids.clear()
        
       
        for line_id in self.propeller_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.physics_client)
            except:
                pass
        self.propeller_line_ids.clear()
        
       
        self._update_visual_elements()
        
        return self._get_obs(), {}
    
    def step(self, action):
        #same physics as training environment
        observation, reward, terminated, truncated, info = super().step(action)
        self._update_visual_elements() 
        time.sleep(0.07)
        return observation, reward, terminated, truncated, info
    
    def close(self):
        #Clean up PyBullet
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def main():
    MODEL_PATH = "quad_slalom_ppo.zip"  
    
    print("Setting up optimized PyBullet environment...")
    env = QuadRotorSlalomVis(render_mode='human')
    
    try:
        print(f"Loading trained model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Please check the path.")
        env.close()
        return
    
    print("Starting optimized live visualization...")
    print("Performance improvements:")
    print("- Simplified geometry (wireframe gates, single drone body)")
    print("- Reduced visual update frequency")
    print("- Fewer trail points and debug lines")
    print("- Optimized camera updates")
    print("- Reduced physics solver iterations")
    
    num_episodes = 0
    while True:
        num_episodes += 1
        print(f"\n--- Episode {num_episodes} ---")
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        start_time = time.time()
        
        while not (terminated or truncated):
            
            action, _states = model.predict(obs, deterministic=True)
           
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
          
            if step_count % 100 == 0:
                fps = step_count / (time.time() - start_time)
                print(f"Step {step_count}: Pos=({env.quad_pos[0]:.2f}, {env.quad_pos[1]:.2f}, {env.quad_pos[2]:.2f}), "
                      f"Target Gate: {env.target_gate_index + 1}/{env.num_gates}, "
                      f"Reward: {episode_reward:.2f}, FPS: {fps:.1f}")
        
        
        end_time = time.time()
        avg_fps = step_count / (end_time - start_time)
        
        print(f"Episode {num_episodes} finished!")
        print(f"Final Reward: {episode_reward:.2f}")
        print(f"Gates Passed: {env.target_gate_index}/{env.num_gates}")
        print(f"Steps: {step_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        

        time.sleep(1)
        
      
        if num_episodes >= 10:
            print("Completed 10 episodes. Exiting...")
            break
    
    env.close()
    print("Optimized visualization finished!")


if __name__ == "__main__":
    main()