import numpy as np
import pdb

class system(object):
	"""docstring for system"""

	def __init__(self, x0, dt):
		self.x = [x0]
		self.u = []
		self.w = []
		self.x0 = x0
		self.dt = dt

	def applyInput(self, ut):
		self.u.append(ut)

        # dynamic bicycle model
		xt = self.x[-1]
		m = 1.98
		lf = 0.125
		lr = 0.125
		Iz = 0.024
		Df = 0.8 * m * 9.81 / 2.0
		Cf = 1.25
		Bf = 1.0
		Dr = 0.8 * m * 9.81 / 2.0
		Cr = 1.25
		Br = 1.0

		# tire split angle
		alpha_f = ut[0] - np.arctan2(xt[1] + lf * xt[2], xt[0])
		alpha_r = - np.arctan2(xt[1] - lf * xt[2], xt[0])

		# lateral force at front and rear tire
		Fyf = Df * np.sin(Cf * np.arctan(Bf * alpha_f))
		Fyr = Dr * np.sin(Cr * np.arctan(Br * alpha_r))

		vx_next = xt[0] + self.dt * (ut[1] - 1 / m * Fyf * np.sin(ut[0]) + xt[2] * xt[1])
		vy_next = xt[1] + self.dt * (1 / m * (Fyf * np.cos(ut[0]) + Fyr) - xt[2] * xt[0])
		phi_next = xt[2] + self.dt * (1 / Iz * (lf * Fyf * np.cos(ut[0]) - lr * Fyr))
		ephi_next = xt[3] + self.dt * (xt[2])
		s_next = xt[4] + self.dt * (xt[0] * np.cos(xt[3]) - xt[1] * np.sin(xt[3]))
		ey_next = xt[5] + self.dt * (xt[0] * np.sin(xt[3]) + xt[1] * np.cos(xt[3]))

		state_next = [vx_next, vy_next, phi_next, ephi_next, s_next, ey_next]

		self.x.append(state_next)

	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []