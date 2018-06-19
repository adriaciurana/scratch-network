from .loss import Loss
import numpy as np
class SoftmaxCrossEntropy(Loss):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		super(SoftmaxCrossEntropy, self).__init__(node, params=params)
	
	def computeSize(self):
		super(SoftmaxCrossEntropy, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(SoftmaxCrossEntropy, self).forward(inputs)

		"""z = inputs[0]
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis] # necessary step to do broadcasting
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis] # dito
		self.values.input = e_x
		self.values.sum = div
		return e_x / div"""
		pred, true = inputs
		probs = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
		probs /= np.sum(probs, axis=-1, keepdims=True)
		self.values.probs = probs
		self.values.true = np.argmax(true, axis=-1)
		return -np.sum(np.log(probs[np.arange(true.shape[0]), self.values.true] + 1e-100)) / true.shape[0]	
		
	def derivatives(self, doutput):
		dx = self.values.probs.copy()
		dx[np.arange(self.values.true.shape[0]), self.values.true] -= 1
		return dx
		