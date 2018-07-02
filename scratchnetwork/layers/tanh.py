from .layer import Layer
import numpy as np
class Tanh(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		params['number_of_inputs'] = 1
		super(Tanh, self).__init__(node, params=params)
	
	def computeSize(self):
		super(Tanh, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Tanh, self).forward(inputs)
		input = inputs[0]

		self.values.input = np.tanh(inputs[0])
		return self.values.input

	def derivatives(self, doutput):
		dx = doutput*(1 - (self.values.input**2))
		return dx