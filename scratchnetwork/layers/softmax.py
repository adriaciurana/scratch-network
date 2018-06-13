from .layer import Layer
import numpy as np
class Softmax(Layer):
	def __init__(self, node, params={}):
		params['number_of_inputs'] = 1
		super(Softmax, self).__init__(node, params)
	
	def computeSize(self):
		super(Softmax, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Softmax, self).forward(inputs)
		input = np.exp(inputs[0])
		self.values.input = input
		self.values.sum = np.sum(np.reshape(input, [-1] + list(self.in_size_flatten[0])), axis=-1)
		input /= self.values.sum
		return input

	def derivatives(self, doutput):
		partial = self.values.input*(self.values.sum - self.values.input)/(self.values.input**2)
		return doutput*partial