from .loss import Loss
import numpy as np
class SoftmaxCrossEntropy(Loss):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		super(SoftmaxCrossEntropy, self).__init__(node, params=params)
	
	def computeSize(self):
		super(SoftmaxCrossEntropy, self).computeSize()
		return tuple([1])

	def forward(self, inputs):
		super(SoftmaxCrossEntropy, self).forward(inputs)

		pred, true = inputs
		probs = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
		probs /= np.sum(probs, axis=-1, keepdims=True)
		self.values.probs = probs
		self.values.true = np.int64(true.flatten()) #np.argmax(true, axis=-1)
		return -np.mean(np.log(probs[np.arange(self.values.true.shape[0]), self.values.true] + 1e-100))
		
	def derivatives(self, doutput):
		dx = self.values.probs.copy()
		dx[np.arange(self.values.true.shape[0]), self.values.true] -= 1
		return dx
		