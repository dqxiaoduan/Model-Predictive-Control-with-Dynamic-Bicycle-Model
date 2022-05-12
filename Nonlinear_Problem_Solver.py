import time

from casadi import *
from numpy import *


class nonlinearproblemsolver(object):
    # parameters initiation
    def __init__(self, horizon, Q, R, Qf, target, delta_t, bx, bu):
        # Define variables
        self.horizon = horizon
        self.states_number = Q.shape[1]
        self.input_number = R.shape[1]
        self.bx = bx
        self.bu = bu
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.target = target
        self.delta_t = delta_t
        self.bx = bx
        self.bu = bu
        self.finitetimeoptimalcontrolproblem()
        self.timetosolve = []

    def solve(self, x_initial, verbose=False):
        # set states and input box constraints
        self.lower_box_constraints = x_initial.tolist() + (-self.bx).tolist() * (self.horizon) + (-self.bu).tolist() * self.horizon
        self.upper_box_constraints = x_initial.tolist() + (self.bx).tolist() * (self.horizon) + (self.bu).tolist() * self.horizon
        # record the time to solve this nonlinear problem
        start_time = time.time()
        solution = self.solver(lbx=self.lower_box_constraints, ubx=self.upper_box_constraints, lbg=self.lower_bound_of_inequality_constraint, ubg=self.upper_bound_of_inequality_constraint)
        end_time = time.time()
        duration = end_time - start_time
        self.timetosolve.append(duration)

        # check if there exists a feasible solution
        if (self.solver.stats()['success']):
            self.feasible = 1
            x = solution["x"]
            self.costofnlpsolver = solution["f"]
            self.predicticted_states = np.array(x[0:(self.horizon + 1) * self.states_number].reshape((self.states_number, self.horizon + 1))).T
            self.predicticted_inputs = np.array(x[(self.horizon + 1) * self.states_number:((self.horizon + 1) * self.states_number + self.input_number * self.horizon)].reshape((self.input_number, self.horizon))).T
            self.mpcInput = self.predicticted_inputs[0][0]
            print("Predicticted states:")
            print(self.predicticted_states)
            print("Predicticted inputs:")
            print(self.predicticted_inputs)
            print("Cost of the nlp solver:")
            print(self.costofnlpsolver)
            print("CASADI Solver Time: ", duration, " s.")
        else:
            self.predicticted_states = np.zeros((self.horizon + 1, self.states_number))
            self.predicticted_inputs = np.zeros((self.horizon, self.input_number))
            self.mpcInput = []
            self.feasible = 0
            print("Unfeasible solution")
        return self.predicticted_inputs[0]

    def finitetimeoptimalcontrolproblem(self):
        states_number= self.states_number
        input_number = self.input_number
        # define the CASADI solver variables
        X = SX.sym('X', states_number * (self.horizon + 1))
        U = SX.sym('U', input_number * self.horizon)
        # define dynamic constraints used by the CASADI solver
        self.dynamic_constraint = []
        for i in range(0, self.horizon):
            X_next = self.nonlineardynamicbicyclemodel(X[states_number * i:states_number * (i + 1)], U[input_number * i:input_number * (i + 1)])
            for j in range(0, self.states_number):
                self.dynamic_constraint = vertcat(self.dynamic_constraint, X_next[j] - X[states_number * (i + 1) + j])
        # define cost used by the CASADI solver
        self.cost = 0
        for i in range(0, self.horizon):
            self.cost = self.cost + (X[states_number * i:states_number * (i + 1)] - self.target).T @ self.Q @ (X[states_number * i:states_number * (i + 1)] - self.target)
            self.cost = self.cost + U[input_number * i:input_number * (i + 1)].T @ self.R @ U[input_number * i:input_number * (i + 1)]
        self.cost = self.cost + (X[states_number * self.horizon:states_number * (self.horizon + 1)] - self.target).T @ self.Qf @ (X[states_number * self.horizon:states_number * (self.horizon + 1)] - self.target)
        # setup the CASADI.ipopt(interior point optimizer
        opts = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        nlp = {'x': vertcat(X, U), 'f': self.cost, 'g': self.dynamic_constraint}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        # Set lower and upper bound of inequality constraint to zeros to force n*N state dynamics
        self.lower_bound_of_inequality_constraint = [0] * (states_number * self.horizon)
        self.upper_bound_of_inequality_constraint = [0] * (states_number * self.horizon)

    def nonlineardynamicbicyclemodel(self, x, u):
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
        # 2 inputs, the first is acceleration, the second is steering
        acceleration = u[0]
        steering = u[1]
        # here use a tiny penalty to make the Jacobian calculation in the CASADI solver correct(force denominator not 0)
        x[3] = x[3] + 0.000001
        # Slip angles for the dynamic bicycle model
        alphaf = steering - np.arctan2(x[4] + lf * x[2], x[3])
        alphar = - np.arctan2(x[4] - lf * x[2], x[3])
        # the lateral forces in the body frame for front and rear wheels
        Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
        Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))
        # Equations for 6 states in the dynamic bicycle model
        x_position = x[0] + self.delta_t * (x[3] * np.cos(x[2]) - x[4] * np.sin(x[2]))
        y_position = x[1] + self.delta_t * (x[3] * np.sin(x[2]) + x[4] * np.cos(x[2]))
        theta = x[2] + self.delta_t * x[5]
        x_velocity = x[3] + self.delta_t * (acceleration - 1 / mass * Ffy * np.sin(steering) + x[4] * x[5])
        y_velocity = x[4] + self.delta_t * (1 / mass * (Ffy * np.cos(steering) + Fry) - x[3] * x[5])
        yaw = x[5] + self.delta_t * (1 / Iz * (lf * Ffy * np.cos(steering) - Fry * lr))
        # return updated/calculated 6 states
        states = [x_position, y_position, theta, x_velocity, y_velocity, yaw]
        return states
