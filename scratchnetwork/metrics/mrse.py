from .metric import Metric
import numpy as np
class MRSE(Metric):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		super(MRSE, self).__init__(node, params=params)


	def forward(self, inputs):
		super(MRSE, self).forward(inputs)
		pred, true = inputs
		out = pred - true
		return np.sqrt(np.mean((out**2).flatten(), axis=0))
