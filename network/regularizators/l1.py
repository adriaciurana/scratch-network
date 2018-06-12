from .regularization import Regularization
import numpy as np
class L1(Regularization):
	def __init__(self, lambda_value):
		super(L1, Regularization).__init__(lambda_value)
	
	def function(self, data):
		return np.dot(self.lambda_value, np.sign(data))
