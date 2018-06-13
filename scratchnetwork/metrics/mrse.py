from .metric import Metric
import numpy as np
class MRSE(Metric):
	def __init__(self, node, params={}):
		super(MRSE, self).__init__(node, params)


	def forward(self, inputs):
		super(MRSE, self).forward(inputs)
		pred, true = inputs
		out = np.reshape(pred, [pred.shape[0], -1]) - np.reshape(true, [true.shape[0], -1])
		return np.sqrt(np.mean((out**2).flatten(), axis=0))
