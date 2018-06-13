from .layer import Layer
import numpy as np
class ReLU(Layer):
	def __init__(self, node, params={}):
		params['number_of_inputs'] = 1
		super(ReLU, self).__init__(node, params)
	
	def computeSize(self):
		super(ReLU, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(ReLU, self).forward(inputs)
		input = inputs[0]
		input = abs(input) * (input > 0)
		self.values.input = input
		return input

	def derivatives(self, doutput):
		partial = self.values.input > 0
		return doutput*partial