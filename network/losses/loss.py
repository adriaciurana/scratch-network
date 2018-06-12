from ..layers.layer import Layer
import numpy as np
class Loss(Layer):
	def __init__(self, node, params= {}):
		params['same_input_shape'] = True
		params['compute_forward_in_prediction'] = False
		super(Loss, self).__init__(node, params)

	def computeSize(self):
		return tuple([1])
		
	def forward(self, inputs):
		super(Loss, self).forward(inputs)
		pred, true = inputs
		out = np.reshape(pred, [pred.shape[0], -1]) - np.reshape(true, [true.shape[0], -1])
		self.values.out = out
		return 0.5*np.mean(out**2, axis=0)

	def backward(self, doutput=None):
		return self.values.out

	def correctWeights(self, doutput):
		pass