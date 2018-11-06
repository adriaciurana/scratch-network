from .regularizator import Regularizator
import numpy as np
class L1(Regularizator):
	def __init__(self, lambda_value=0.01):
		super(L1, self).__init__(lambda_value)
	
	def function(self, data):
		def sign(data):
			return 1.*(data >= 0) - 1.*(data < 0)
		
		if self.lambda_value == 0:
			return 0
		
		return self.lambda_value * sign(data)
