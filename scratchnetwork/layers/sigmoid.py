from .layer import Layer
import numpy as np
class Sigmoid(Layer):
	def __init__(self, node, params={}):
		params['number_of_inputs'] = 1
		super(Sigmoid, self).__init__(node, params=params)
	
	def computeSize(self):
		super(Sigmoid, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Sigmoid, self).forward(inputs)
		input = inputs[0]
		input = 1./(1. + np.exp(- input))
		self.values.input = input
		return input

	def derivatives(self, doutput):
		partial = self.values.input*(1 - self.values.input)
		return doutput*partial