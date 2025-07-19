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
import pybullet as p
import pybullet_data
from quadrotor_dive import QuadRotorDive 

#Most of the functions are explained in quadrotor_dive script

# Quadrotor Dive Environment enhanced for PyBullet Visualization
class QuadRotorDiveVis(QuadRotorDive):

    def __init__(self, render_mode=None):
        super().__init__(render_mode)

        # Visualization Parameters 
        self.update_visual_every_n_steps = 2
        self.trail_update_interval = 5
        self.camera_update_interval = 10
        self.max_trail_points = 50

        # PyBullet Setup 
        self.physics_client = None
        self.quad_id = None
        self.propeller_visual_ids = []
        self.propeller_line_ids = []
        self.trail_points = []
        self.trail_line_ids = []
        self.target_zone_id = None
        self.ground_id = None
        self.step_counter = 0
        self.last_camera_pos = np.zeros(3)
        self.rotor_positions_body = np.array([
            [self.L/np.sqrt(2), self.L/np.sqrt(2), 0.1],
            [self.L/np.sqrt(2), -self.L/np.sqrt(2), 0.1],
            [-self.L/np.sqrt(2), self.L/np.sqrt(2), 0.1],
            [-self.L/np.sqrt(2), -self.L/np.sqrt(2), 0.1]
        ], dtype=np.float32)

        if self.render_mode in ['human', 'rgb_array']:
            self._setup_pybullet()

    def _setup_pybullet(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(self.dt, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(numSolverIterations=5, physicsClientId=self.physics_client)

        ground_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.1],
            rgbaColor=[0.4, 0.4, 0.2, 1.0],
            physicsClientId=self.physics_client
        )
        self.ground_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=ground_visual,
            basePosition=[0, 0, -0.1],
            physicsClientId=self.physics_client
        )

        target_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.success_xy_threshold, self.success_xy_threshold, 0.01],
            rgbaColor=[0.0, 1.0, 1.0, 0.3],
            physicsClientId=self.physics_client
        )
        self.target_zone_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[self.dive_target_xy[0], self.dive_target_xy[1], self.dive_target_z],
            physicsClientId=self.physics_client
        )

        self._create_simplified_quadrotor()
        self._set_camera()

    def _create_simplified_quadrotor(self):
        body_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.05], physicsClientId=self.physics_client)
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.15, 0.05],
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

    def _set_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=12,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, self.initial_altitude_mean],
            physicsClientId=self.physics_client
        )

    def _update_visual_elements(self):
        self.step_counter += 1
        pybullet_quat = [self.quad_quat[1], self.quad_quat[2], self.quad_quat[3], self.quad_quat[0]]
        p.resetBasePositionAndOrientation(
            self.quad_id,
            self.quad_pos,
            pybullet_quat,
            physicsClientId=self.physics_client
        )

        if self.step_counter % self.update_visual_every_n_steps == 0:
            self._update_propeller_indicators()

        if self.step_counter % self.trail_update_interval == 0:
            self._update_trail()

        if self.step_counter % self.camera_update_interval == 0:
            self._update_camera()

    def _update_propeller_indicators(self):
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
            angle = (self.step_counter * 0.5) % (2 * np.pi)
            prop_quat = [0, 0, np.sin(angle/2), np.cos(angle/2)]
            p.resetBasePositionAndOrientation(
                prop_id,
                pos_world,
                prop_quat,
                physicsClientId=self.physics_client
            )

    def _update_trail(self):
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
                line_id = p.addUserDebugLine(
                    self.trail_points[i],
                    self.trail_points[i + step],
                    lineColorRGB=[0, 1, 0],
                    lineWidth=2,
                    physicsClientId=self.physics_client
                )
                self.trail_line_ids.append(line_id)

    def _update_camera(self):
        pos_diff = np.linalg.norm(self.quad_pos - self.last_camera_pos)
        if pos_diff > 2.0:
            p.resetDebugVisualizerCamera(
                cameraDistance=12,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=self.quad_pos,
                physicsClientId=self.physics_client
            )
            self.last_camera_pos = self.quad_pos.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.step_counter = 0
        self.trail_points = [self.quad_pos.copy()]
        self.last_camera_pos = self.quad_pos.copy()

        if self.render_mode in ['human', 'rgb_array']:
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

        return self._get_obs(), self._get_info()

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        if self.render_mode in ['human', 'rgb_array']:
            self._update_visual_elements()


        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            self._update_visual_elements()
            time.sleep(0.02)
            return None
        elif self.render_mode == 'rgb_array':
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.quad_pos,
                distance=12,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.physics_client
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.physics_client
            )
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
            return rgb_array
        return None

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
        plt.close('all')

if __name__ == "__main__":
    LOAD_MODEL_PATH = "quad_dive_ppo.zip"

    # Check if model exists
    if not os.path.exists(LOAD_MODEL_PATH):
        print(f"Error: Model file not found at {LOAD_MODEL_PATH}. Please train the model first.")
        exit(1)
    
    print("Visualizing learned policy for Dive task...")
    
    # Create a single GUI environment for visualization
    vis_env = QuadRotorDiveVis(render_mode='human')
    
    # Load the model (create a dummy env for loading)
    dummy_env = DummyVecEnv([lambda: QuadRotorDiveVis(render_mode=None)])
    
    print(f"Loading model from {LOAD_MODEL_PATH}...")
    model = PPO.load(LOAD_MODEL_PATH, env=dummy_env)
    dummy_env.close()
    
    # Run evaluation episodes
    total_reward_eval = 0
    num_eval_episodes = 5
    successful_dives = 0

    for episode in range(num_eval_episodes):
        obs, info = vis_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        print(f"\n--- Evaluation Episode {episode + 1} ---")
        initial_pos = obs[:3]
        print(f"Starting at: X={initial_pos[0]:.2f}, Y={initial_pos[1]:.2f}, Z={initial_pos[2]:.2f}")

        for step_num in range(vis_env.max_steps_per_episode):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vis_env.step(action)
            episode_reward += reward
            vis_env.render()  # This will update the GUI

            if terminated or truncated:
                break

        total_reward_eval += episode_reward
        final_pos = info.get("current_pos", obs[:3])
        final_alt = info.get("altitude", obs[2])
        dist_z = info.get("dist_to_target_z", float('inf'))
        dist_xy = info.get("dist_to_target_xy", float('inf'))
        final_vz = info.get("current_vel", [0, 0, 0])[2] if "current_vel" in info else obs[5]

        print(f"Episode {episode + 1} Finished. Steps: {step_num+1}")
        print(f"  Final Position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}, Z={final_alt:.2f}")
        print(f"  Vz: {final_vz:.2f} m/s")
        print(f"  Distance to Target Z: {dist_z:.2f} m, XY: {dist_xy:.2f} m")
        print(f"  Episode Reward: {episode_reward:.2f}")

        if dist_z < vis_env.success_z_threshold and \
           dist_xy < vis_env.success_xy_threshold and \
           abs(final_vz) < vis_env.success_vz_threshold and \
           not terminated:
            successful_dives += 1
            print("  Dive Considered SUCCESSFUL!")
        elif terminated:
            print("  Dive Terminated (e.g., crash, flip).")
        else:
            print("  Dive Incomplete or Missed Target.")

        print("Pausing to view final trajectory...")
        time.sleep(2)

    print(f"\n--- Dive Visualization Finished ---")
    print(f"Average Reward over {num_eval_episodes} episodes: {total_reward_eval / num_eval_episodes:.2f}")
    print(f"Successful dives: {successful_dives}/{num_eval_episodes}")

    vis_env.close()

    print("Dive script finished.")