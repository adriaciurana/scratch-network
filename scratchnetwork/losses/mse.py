from .loss import Loss
import numpy as np
class MSE(Loss):
	def __init__(self, node, params={}):
		super(MSE, self).__init__(node, params=params)
		
	def forward(self, inputs):
		super(MSE, self).forward(inputs)
		pred, true = inputs
		out = pred - true
		self.values.out = out
		out = np.reshape(out, [true.shape[0], -1])
		return 0.5*np.mean((out**2).flatten(), axis=0)

	def derivatives(self, doutput=None):
		return self.values.out