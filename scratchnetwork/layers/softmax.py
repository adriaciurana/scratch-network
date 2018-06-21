from .layer import Layer
import numpy as np
from .cython import softmax
class Softmax(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		params['number_of_inputs'] = 1
		super(Softmax, self).__init__(node, params=params)
	
	def computeSize(self):
		super(Softmax, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Softmax, self).forward(inputs)
		
		self.values.out = softmax.nb_forward(inputs[0])
		return self.values.out
		
	def derivatives(self, doutput):
		#https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights
		dx = softmax.nb_derivatives(doutput, self.values.out)
		return dx
		