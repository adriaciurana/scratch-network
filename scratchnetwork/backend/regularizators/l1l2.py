from .regularizator import Regularizator
import numpy as np
from .l1 import L1
from .l2 import L2
class L1L2(L1, L2):
	def __init__(self, lambda_valueL1=0.01, lambda_valueL2=0.01):
		L1.__init__(lambda_valueL1)
		L2.__init__(lambda_valueL2)
	
	def function(self, data):
		return L1.function(data) + L2.function(data)