from .optimizer import Optimizer
import numpy as np
class SGD(Optimizer):
	def __init__(self, lr=1e-1, mu=0.9, clip=1):
		self.lr = lr
		self.mu = mu
		self.clip = clip
	
	def step(self, dweight, iweight_1):
		
		if self.clip is None:
			coef = 1
		else:
			dweight_flatten = dweight.flatten()
			dweight_norm = np.linalg.norm(dweight_flatten, 2)
			coef = self.clip/max(dweight_norm, self.clip)

		return - self.lr*dweight*coef + self.mu*iweight_1
