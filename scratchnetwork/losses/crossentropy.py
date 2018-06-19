from .loss import Loss
import numpy as np
class CrossEntropy(Loss):
	def __init__(self, node, params={}):
		super(CrossEntropy, self).__init__(node, params=params)
		
	def forward(self, inputs):
		super(CrossEntropy, self).forward(inputs)

		pred, true = inputs
		self.values.pred = pred + 1e-100
		self.values.true = true
		out = - self.values.true*np.log(self.values.pred) #- (1 - true)*np.log(1 - pred)
		return np.mean(np.sum(out, axis=-1), axis=0)

	def derivatives(self, doutput=None):
		dx = - (self.values.true/self.values.pred)
		return dx #- ((1 - self.values.true)/(1 - self.values.pred))