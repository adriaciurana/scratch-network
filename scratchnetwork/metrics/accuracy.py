from .metric import Metric
import numpy as np
class Accuracy(Metric):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		super(Accuracy, self).__init__(node, params=params)


	def forward(self, inputs):
		super(Accuracy, self).forward(inputs)

		pred, true = inputs
		return np.sum(pred == true) / pred.shape[0]
