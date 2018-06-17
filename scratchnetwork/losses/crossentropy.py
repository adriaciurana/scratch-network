from .loss import Loss
import numpy as np
class CrossEntropy(Loss):
	def __init__(self, node, params={}):
		super(CrossEntropy, self).__init__(node, params=params)
		
	def forward(self, inputs):
		super(CrossEntropy, self).forward(inputs)

		pred, true = inputs
		self.values.pred = pred
		self.values.true = true
		out = - true*np.log(pred) - (1 - true)*np.log(1 - pred)
		return np.mean(np.sum(out, axis=-1), axis=0)

	def derivatives(self, doutput=None):
		return - (self.values.true/self.values.pred) - ((1 - self.values.true)/(1 - self.values.pred))