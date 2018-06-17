from .layer import Layer
import numpy as np
from .cython import softmax
class Softmax(Layer):
	def __init__(self, node, params={}):
		params['number_of_inputs'] = 1
		super(Softmax, self).__init__(node, params)
	
	def computeSize(self):
		super(Softmax, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Softmax, self).forward(inputs)

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
		self.values.o = softmax.nb_forward(inputs[0])
		return self.values.o
		
	def derivatives(self, doutput):
		#https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights
		dx = softmax.nb_derivatives(doutput, self.values.o)
		return dx
		