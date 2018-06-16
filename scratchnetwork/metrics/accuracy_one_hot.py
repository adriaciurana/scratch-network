from .metric import Metric
import numpy as np
class Accuracy(Metric):
	def __init__(self, node, params={}):
		super(Accuracy, self).__init__(node, params)


	def forward(self, inputs):
		super(Accuracy, self).forward(inputs)
		pred, true = inputs
		pred = np.argmax(pred, axis=-1)
		true = np.argmax(true, axis=-1)

		return np.sum(pred == true) / pred.shape[0]
