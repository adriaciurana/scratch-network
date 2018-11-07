from .layer import Layer
import numpy as np
class Reshape(Layer):
	def __init__(self, node, shape, params=None):
		if params is None:
			params = {}
		params['number_of_inputs'] = 1
		super(Reshape, self).__init__(node, params=params)
		self.shape = shape
	
	def computeSize(self):
		super(Reshape, self).computeSize()
		return (self.shape, )

	def forward(self, inputs):
		super(Reshape, self).forward(inputs)
		return inputs[0].reshape([-1, self.shape])

	def derivatives(self, doutput):
		return doutput.reshape([-1] + self.in_size[0])
