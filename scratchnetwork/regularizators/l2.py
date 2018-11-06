from .regularizator import Regularizator
import numpy as np
class L2(Regularizator):
	def __init__(self, lambda_value=0.01):
		super(L2, self).__init__(lambda_value)
	
	def function(self, data):
		if self.lambda_value == 0:
			return 0
		return self.lambda_value * data
