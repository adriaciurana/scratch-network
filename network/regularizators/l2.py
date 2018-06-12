from .regularization import Regularization
import numpy as np
class L2(Regularization):
	def __init__(self, lambda_value):
		super(L2, Regularization).__init__(lambda_value)
	
	def function(self, data):
		return np.dot(self.lambda_value * 2, data)
