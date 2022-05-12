import matplotlib.pyplot as plt
import numpy as np

from Nonlinear_Problem_Solver import nonlinearproblemsolver
from Simulator_Code import dynamicbicyclemodel

# Initialize Parameters for both simulator and the NLP solver
horizon = 25
number_of_states = 6
number_of_inputs = 2
x_initial_dynamic = np.array([0, 0, 0, 3.5, 0, 0])# np.zeros(6)
delta_t = 0.12
simulator = dynamicbicyclemodel(x_initial_dynamic, delta_t)
simulation_time = 25
goal = np.array([10, 10, np.pi / 2, 3.5, 0, 0])
R_dynamic = 1 * np.eye(number_of_inputs)
Q_dynamic = 1 * np.eye(number_of_states)
Qf_dynamic = np.diag([11.8, 2.0, 50.0, 280.0, 100.0, 1000.0])
bx_dynamic = np.array([15, 15, 15, 15, 15, 15])
bu_dynamic = np.array([10, 0.5])

# run the simulation in the simulator and solve the nonlinear problem using the CASADI-ipopt solver
nlp_dynamic = nonlinearproblemsolver(horizon, Q_dynamic, R_dynamic, Qf_dynamic, goal, delta_t, bx_dynamic, bu_dynamic)
ut_dynamic = nlp_dynamic.solve(x_initial_dynamic)
simulator.reset_initialcondition()
predicted_states_dynamic = []
predicted_inputs_dynamic = []
cost_nlpsolver_dynamic = []

for t in range(0, simulation_time):
    xt_dynamic = simulator.x[-1]
    ut_dynamic = nlp_dynamic.solve(xt_dynamic)
    predicted_states_dynamic.append(nlp_dynamic.predicticted_states)
    predicted_inputs_dynamic.append(nlp_dynamic.predicticted_inputs)
    cost_nlpsolver_dynamic.append(nlp_dynamic.costofnlpsolver)
    simulator.applytheinput(ut_dynamic)

# states and inputs for the closed-loop simulation
simulator_states_dynamic = np.array(simulator.x)
simulator_inputs_dynamic = np.array(simulator.u)

# plotting for the actual cost, The
# actual control cost is calculated by 𝐽(𝑥, 𝑢) on the actual x and u trajectories, i.e.,
# take the actual closed-loop trajectories {𝑥(0), 𝑥(1), ···} and {𝑢(0), 𝑢(1), ···}.
# Note that smaller actual cost means better performance if the solutions for all the
# time steps are feasible.
cost_accumulated = 0
cost_actual = []
for i in range(0, horizon):
    cost_accumulated = (simulator_states_dynamic[i] - goal).T @ Q_dynamic @ (simulator_states_dynamic[i] - goal)
    cost_accumulated += simulator_inputs_dynamic[i].T @ R_dynamic @ simulator_inputs_dynamic[i]
    cost_accumulated += (simulator_states_dynamic[i] - goal).T @ Qf_dynamic @ (simulator_states_dynamic[i] - goal) + \
                        simulator_inputs_dynamic[i].T @ R_dynamic @ simulator_inputs_dynamic[i]
    cost_actual.append(cost_accumulated)
print("Actual cost:", sum(cost_actual))

# The actual loss can be used to measure the fitting error between the NLP-predicted
# results and the simulated results, which is the square error for the actual
# states and inputs
loss_x_position = 0
loss_x_accu = []
loss_y_position = 0
loss_y_accu = []
loss_heading_angle = 0
loss_heading_angle_accu = []
for i in range(0, horizon):
    lost_x_position = (simulator_states_dynamic[i, 0] - predicted_states_dynamic[0][i, 0]) ** 2
    loss_x_accu.append(lost_x_position)
    lost_y_position = (simulator_states_dynamic[i, 1] - predicted_states_dynamic[0][i, 1]) ** 2
    loss_y_accu.append(lost_y_position)
    loss_heading_angle = (simulator_states_dynamic[i, 2] - predicted_states_dynamic[0][i, 2]) ** 2
    loss_heading_angle_accu.append(loss_heading_angle)

# Calculate the solving time for the NLP solver
solvetime_dynamic = sum(nlp_dynamic.timetosolve) / len(nlp_dynamic.timetosolve)
print("Solving time for the dynamic model using the NLP solver:", solvetime_dynamic)

# plotting for the evolution of the predicted trajectory from the NLP solver
# for time in [0, 10]:
#     plt.figure()
#     plt.plot(predicted_states_dynamic[time][:, 0], predicted_states_dynamic[time][:, 1], '--.b',
#              label="Simulated trajectory using NLP-aided MPC at time $t = $" + str(time))
#     plt.plot(predicted_states_dynamic[time][0, 0], predicted_states_dynamic[time][0, 1], 'ok',
#              label="$x_t$ at time $t = $" + str(time))
#     plt.xlim(-1, 15)
#     plt.ylim(-1, 15)
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.legend()
    # plt.show()

# plotting for the comparison between the trajectory from the NLP-solver-solved MPC and the trajectory from the simulator for the entire time duration of the MPC simulation
# plt.figure()
# for t in range(0, simulation_time):
#     if t == 0:
#         plt.plot(predicted_states_dynamic[t][:, 0], predicted_states_dynamic[t][:, 1], '--.b',
#                  label='Simulated trajectory using NLP-aided MPC')
#     else:
#         plt.plot(predicted_states_dynamic[t][:, 0], predicted_states_dynamic[t][:, 1], '--.b')
# plt.plot(simulator_states_dynamic[:, 0], simulator_states_dynamic[:, 1], '-*r', label="Closed-loop trajectory")
# plt.xlim(-1, 15)
# plt.ylim(-1, 15)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.legend()
# plt.show()

# plotting for the comparison between the trajectory from the NLP-solver-solved MPC and the trajectory from the simulator at the start time of the MPC simulation
# plt.figure()
# plt.plot(predicted_states_dynamic[0][:, 0], predicted_states_dynamic[0][:, 1], '--.b',
#          label='Simulated trajectory using NLP-aided MPC')
# plt.plot(simulator_states_dynamic[:, 0], simulator_states_dynamic[:, 1], '-*r', label="Closed-loop trajectory")
# plt.xlim(-1, 15)
# plt.ylim(-1, 15)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.legend()
# plt.show()

# plotting for the comparison of the x-direction velocity from the NLP-solver-solved MPC and the x-direction velocity from the simulator
# plt.figure()
# plt.plot(predicted_states_dynamic[0][:, 3], '-*r', label='NLP performance')
# plt.plot(simulator_states_dynamic[:, 3], 'ok', label='Closed-loop performance')
# plt.xlabel('Time')
# plt.ylabel('Velocity of the x-axis')
# plt.legend()
# plt.show()

# plotting for the comparison of the heading angle from the NLP-solver-solved MPC and the heading angle from the simulator
# plt.figure()
# plt.plot(predicted_states_dynamic[0][:, 2], '-*r', label='NLP performance')
# plt.plot(simulator_states_dynamic[:, 2], 'ok', label='Closed-loop performance')
# plt.xlabel('Time')
# plt.ylabel('Heading angle(rad)')
# plt.legend()
# plt.show()

# plotting for the iteration cost from the CASADI solver
# plt.figure()
# plt.plot(cost_nlpsolver_dynamic, '-ob')
# plt.xlabel('Time')
# plt.ylabel('Iteration cost')
# plt.legend()
# plt.show()

# plotting of the loss for both x and y position
# plt.figure()
# plt.plot(loss_x_accu, 'o-b', label='X-Position')
# plt.plot(loss_y_accu, '*-r', label='Y-Position')
# plt.plot(loss_heading_angle_accu, '*-g', label='Heading angle')
# plt.xlabel('Time')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

## plotting for MPC-predicted x position vs the closed-loop simulated x-position
# arr = np.array(predicted_states_dynamic)
# arr_1 = np.array(simulator.x)
# plt.figure()
# time = np.linspace(0, 25, 26)
# for t in range(0, simulation_time):
#     if t == 0:
#         plt.plot(predicted_states_dynamic[t][:, 0], '--.b', label='NLP-predicted x-position')
#     else:
#         time_1 = np.linspace(t, 25, 26 - t)
#         time_1 = time_1.tolist()
#         predicted_states_dynamic[t][t, 0] = arr_1[t, 0]
#         plt.plot(time_1, predicted_states_dynamic[t][t:26, 0], '--.b')
# plt.plot(time, arr_1[:, 0], '-*r', label="Close-loop simulated x-position")
# plt.xlabel('$Time$')
# plt.ylabel('$x-position$')
# plt.legend()
# plt.show()

plt.figure()
for t in range(0, simulation_time):
    if t == 0:
        plt.plot(predicted_states_dynamic[t][:, 0], '--.b',
                 label='Simulated x-position using NLP-aided MPC')
    else:
        time = np.arange(t, 26)
        plt.plot(time, predicted_states_dynamic[t][0:26-t, 0], '--.b')
plt.plot(simulator_states_dynamic[:, 0], '-*r', label="Closed-loop x-position")
plt.xlim(-1, 15)
plt.ylim(-1, 15)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()