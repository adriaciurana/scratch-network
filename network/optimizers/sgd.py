from .optimizer import Optimizer
import numpy as np
class SGD(Optimizer):
	def __init__(self, lr=1e-1, clip=1):
		self.lr = lr
		self.clip = clip
	
	def step(self, dweight):
		dweight_flatten = dweight.flatten()
		dweight_norm = np.linalg.norm(dweight_flatten, 2)
		
		if self.clip is None:
			coef = 1
		else:
			coef = self.clip/max(dweight_norm, self.clip)
		return -self.lr*dweight*coef
