from .layer import Layer
import numpy as np
class Flatten(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
		params['number_of_inputs'] = 1
		super(Flatten, self).__init__(node, params=params)
	
	def computeSize(self):
		super(Flatten, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Flatten, self).forward(inputs)
		return inputs[0].reshape([-1, self.in_size_flatten[0]])

	def derivatives(self, doutput):
		return doutput.reshape([-1] + list(self.in_size[0]))