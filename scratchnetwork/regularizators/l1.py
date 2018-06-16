from .regularization import Regularization
import numpy as np
class L1(Regularization):
	def __init__(self, lambda_value):
		super(L1, self).__init__(lambda_value)
	
	def function(self, data):
		return self.lambda_value * np.sign(data)
