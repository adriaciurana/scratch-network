from .loss import Loss
import numpy as np
class CrossEntropy(Loss):
	def __init__(self, node, params= {}):
		super(CrossEntropy, self).__init__(node, params)
		
	def forward(self, inputs):
		super(CrossEntropy, self).forward(inputs)
		pred, true = inputs
		self.values.pred = pred
		self.values.true = true
		out = -true*np.log(pred)
		self.values.out = out
		out = np.reshape(out, [true.shape[0], -1])
		return np.mean((out**2).flatten(), axis=0)

	def derivatives(self, doutput=None):
		return -self.values.true/self.values.pred