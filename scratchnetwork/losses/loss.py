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
		out = pred - true
		self.values.out = out
		out = np.reshape(out, [true.shape[0], -1])
		return 0.5*np.mean((out**2).flatten(), axis=0)

	def derivatives(self, doutput=None):
		return self.values.out

	def correctWeights(self, doutput):
		pass