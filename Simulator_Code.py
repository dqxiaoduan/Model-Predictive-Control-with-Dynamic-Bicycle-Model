import numpy as np
import pdb
import math


# dynamic bicycle model class for the simulator
class dynamicbicyclemodel(object):
    # parameters initiation
    def __init__(self, x_initial, delta_t):
        self.x = [x_initial]
        self.u = []
        self.w = []
        # initial conditions for the 6 states, initial conditions are all zeros
        self.x_initial = x_initial
        # simulation step-size for the simulator
        self.delta_t = delta_t

    # apply input and dynamic bicycle model for the simulator
    def applytheinput(self, u_simulator):
        self.u.append(u_simulator)
        x_simulator = self.x[-1]
        # below are parameters from the Berkeley Autonomous Race Car Platform
        # mass
        mass = 1.98
        # distance from the center of gravity to front and rear axles
        lf = 0.125
        lr = 0.125
        # moment of inertia about the vertical axis passing through the center of gravity
        Iz = 0.024
        # track specific parameters for the tire force curves
        Df = 0.8 * mass * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * mass * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0
        # Slip angles for the dynamic bicycle model
        alphaf = u_simulator[1] - np.arctan2(x_simulator[4] + lf * x_simulator[2], x_simulator[3])
        alphar = - np.arctan2(x_simulator[4] - lf * x_simulator[2], x_simulator[3])
        # the lateral forces in the body frame for front and rear wheels
        Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
        Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))
        # Equations for 6 states in the dynamic bicycle model
        x_position = x_simulator[0] + self.delta_t * (
                x_simulator[3] * np.cos(x_simulator[2]) - x_simulator[4] * np.sin(x_simulator[2]))
        y_position = x_simulator[1] + self.delta_t * (
                x_simulator[3] * np.sin(x_simulator[2]) + x_simulator[4] * np.cos(x_simulator[2]))
        theta = x_simulator[2] + self.delta_t * x_simulator[5]
        x_velocity = x_simulator[3] + self.delta_t * (
                u_simulator[0] - 1 / mass * Ffy * np.sin(u_simulator[1]) + x_simulator[5] * x_simulator[4])
        y_velocity = x_simulator[4] + self.delta_t * (
                1 / mass * (Ffy * np.cos(u_simulator[1]) + Fry) - x_simulator[5] * x_simulator[3])
        yaw = x_simulator[5] + self.delta_t * (1 / Iz * (lf * Ffy * np.cos(u_simulator[1]) - Fry * lr))
        # return updated/calculated 6 states
        states = np.array([x_position, y_position, theta, x_velocity, y_velocity, yaw])
        self.x.append(states)

    def reset_initialcondition(self):
        self.x = [self.x_initial]
        self.u = []
        self.w = []
