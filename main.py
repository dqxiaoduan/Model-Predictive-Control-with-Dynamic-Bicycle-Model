import numpy as np
from utils import system
import pdb
import matplotlib.pyplot as plt
# from ftocp import FTOCP
from nlp import NLP
from nlp import *
from matplotlib import rc
from numpy import linalg as la

# =============================
# Initialize system parameters
x0 = np.array([0.5, 0, 0, 0, 0, 0])
dt = 0.1  # Discretization time
sys = system(x0, dt)  # initialize system object
maxTime = 14 # 50 for kinematic bicycle model
goal = np.array([0, 0, 0, np.pi / 2, 10, 0])  # same goal as the kinematic bicycle model

# Initialize mpc parameters
N = 14 # 50 for kinematic bicycle model
n = 6  # 4 for kinematic bicycle model
d = 2
Q = np.diag([1, 1, 1, 1, 0, 100])
R = np.diag([1, 10])
Qf = np.zeros((n, n))  # kinematic bicycle model: Qf = 1000 * np.eye(n)

# =================================================================
# ======================== Subsection: Nonlinear MPC ==============
# # First solve the nonlinear optimal control problem as a Non-Linear Program (NLP)
printLevel = 1
bx = np.array([0, 0, 0, 0, 0, 1])
bu = np.array([0.5, 10])  # acceleration limit is 10, steering limit is 0.5
nlp = NLP(N, Q, R, Qf, goal, dt, bx, bu, printLevel)
ut = nlp.solve(x0)

sys.reset_IC()  # Reset initial conditions
Cost = []
xPredNLP = []
uPredNLP = []

for t in range(0, maxTime):  # Time loop
    xt = sys.x[-1]
    ut = nlp.solve(xt)
    # Cost.append(nlp.NLPCost)
    xPredNLP.append(nlp.xPred)
    uPredNLP.append(nlp.uPred)
    sys.applyInput(ut)
#
x_cl_nlp = np.array(sys.x)  # closed-loop simulation states
#

# # plot for the animation of the trajectory as time evolute
# for timeToPlot in [0, 20]:
#     plt.figure()
#     plt.plot(xPredNLP[timeToPlot][:, 0], xPredNLP[timeToPlot][:, 1], '--.b',
#              label="Predicted trajectory at time $t = $" + str(timeToPlot))
#     plt.plot(xPredNLP[timeToPlot][0, 0], xPredNLP[timeToPlot][0, 1], 'ok',
#              label="$x_t$ at time $t = $" + str(timeToPlot))
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.xlim(-1, 12)
#     plt.ylim(-1, 10)
#     plt.legend()
#     plt.show()

# # code used to plot trajectory comparison plots
# plt.figure()
# for t in range(0, maxTime):
#     if t == 0:
#         plt.plot(xPredNLP[t][:, 0], xPredNLP[t][:, 1], '--.b', label='Predicted trajectory using NLP solver')
#     else:
#         plt.plot(xPredNLP[t][:, 0], xPredNLP[t][:, 1], '--.b')
# plt.plot(x_cl_nlp[:, 0], x_cl_nlp[:, 1], '-*r', label="Closed-loop simulation trajectory")
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.xlim(-1, 12)
# plt.ylim(-1, 12)
# plt.legend()
# plt.show()

# # used to plot the comparison plt between closed-loop results and simulated results(for states)
# arr = numpy.array(nlp.xPred)
# arr_1 = arr.reshape(1050, 4)
# arr = numpy.array(xPredNLP)
# arr_2 = arr.reshape(210, 6)  # 2550 for N=50
# arr_1 = numpy.array(sys.x)
# print(arr_2)
# print(arr_2.shape)
# arr_3 = np.zeros(51)
# for i in range(51):
#     arr_3[i] = arr_2[i, 0]
# plt.figure()
# time = np.linspace(0, 50, 51)
# for t in range(0, maxTime):
#     if t == 0:
#         plt.plot(xPredNLP[t][:, 0], '--.b', label='NLP-predicted Vx')
#     else:
#         plt.plot(xPredNLP[t][:, 0], '--.b')
# # plt.plot(time, arr_3, '.b', label="NLP-predicted heading angle at time t=0 with horizon N=50")
# plt.plot(time, arr_1[:, 0], '-*r', label="Close-loop simulated Vx")
# plt.xlabel('$Time$')
# plt.ylabel('$Vx$')
# plt.legend()
# plt.show()

# # code used to plot inputs
# arr_4 = np.array(sys.u)
# arr_5 = np.array(uPredNLP)
# arr_6 = arr_5.reshape(196, 2)  # 2500 for N=50
# arr_7 = np.zeros(50)
# for i in range(50):
#     arr_7[i] = arr_6[i, 1]
# plt.figure()
# time = np.linspace(0, 50, 50)
# for t in range(0, maxTime):
#     if t == 0:
#         plt.plot(uPredNLP[t][:, 1], '--.b', label='NLP-predicted input of steering')
#     else:
#         plt.plot(uPredNLP[t][:, 1], '--.b')
# plt.plot(time, arr_7, '.b', label="NLP-predicted input steering")
# plt.plot(time, arr_4[:, 1], '-*r', label="Close-loop simulated input of steering")
# plt.xlabel('$Time$')
# plt.ylabel('$Steering$')
# plt.legend()
# plt.show()

# # code used to plot cost
# plt.figure()
# plt.plot(Cost, '-or')
# plt.xlabel('Time')
# plt.ylabel('Cost')
# plt.show()