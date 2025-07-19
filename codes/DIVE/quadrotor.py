import gymnasium as gym
import numpy as np
from gymnasium import spaces


class QuadRotor(gym.Env):
    """
    Super Class of the quadrotor to be used to train different tasks, implements physics methods to update dynamics and parameters 
    """    
    def __init__(self):
        super().__init__() # inherits from gym.Env 
        
        self.dt = 0.05  # 20Hz simulation
        self.gravity = 9.81 #accelaration of gravity 
        self.mass = 0.5  # kg 
        self.L = 0.15  # distance from rotor to rotor
        
        # Moments of inertia (kg⋅m²)
        self.Ixx = 0.0075
        self.Iyy = 0.0075
        self.Izz = 0.013
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz]).astype(np.float32)
        self.c_torque_thrust_ratio = 0.02 #drag coef/thrust coef
        self.inv_I = np.linalg.inv(self.I)
        self.max_steps_per_episode = 400  
        hover = (self.mass * self.gravity) / 4.0
        self.max_motor_thrust = hover * 3.0  # N per motor, 3*G = mg
        self.min_motor_thrust = 0.0
        self.collision_radius = 0.5

        self.max_velocity = 10.0 #m/s
        self.max_angular_velocity = 20.0 #rad/s
        #actions: forces
        # [F1, F2, F3, F4] 
        low_action = np.array([self.min_motor_thrust] * 4, dtype=np.float32)
        high_action = np.array([self.max_motor_thrust] * 4, dtype=np.float32)
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)
        
    def quaternion_multiply(self, q1, q0):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                         x1*w0 + y1*z0 - z1*y0 + w1*x0,
                         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                         x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float32)

    def quaternion_to_rotation_matrix(self,q):
        #Orientation information
        w, x, y, z = q
        Nq = w*w + x*x + y*y + z*z
        if Nq < np.finfo(np.float32).eps:
            return np.identity(3)
        s = 2.0/Nq
        X = x*s; Y = y*s; Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z
        return np.array(
            [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
             [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
             [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]], dtype=np.float32)

    def _get_rotation_matrix(self, euler):
        #Get rotation matrix from Euler angles
        roll, pitch, yaw = euler # axis x, y and z angles 
        
        # rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        #order of rotation ZYX
        return Rz @ Ry @ Rx

    def _euler_rates_to_body_rates(self, euler, euler_rates):
        #Convert Euler angle rates to body angular velocities
        roll, pitch, yaw = euler
        
        # Transformation matrix
        T = np.array([
            [1, 0, -np.sin(pitch)],
            [0, np.cos(roll), np.cos(pitch)*np.sin(roll)],
            [0, -np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])
        
        return T @ euler_rates

    def _body_rates_to_euler_rates(self, euler, body_rates):
        #Convert body angular velocities to Euler angle rates
        roll, pitch, yaw = euler
        
        # Avoid singularity at pitch = ±90°
        if abs(np.cos(pitch)) < 1e-6:
            # Use approximate transformation near singularity
            T_inv = np.eye(3)
        else:
            # Transformation matrix (inverse of the above)
            T_inv = np.array([
                [1, np.sin(roll)*np.tan(pitch), np.cos(roll)*np.tan(pitch)],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]
            ])
        
        return T_inv @ body_rates


