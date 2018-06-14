from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
class FC(Layer):
	def __init__(self, node, neurons, initializer={'weights': Initializer("normal"), 'bias': Initializer("normal")}, params={}):
		super(FC, self).__init__(node, weights_names=('weights', 'bias'))
		
		self.initializer = initializer
		self.neurons = neurons
	
	def computeSize(self):
		super(FC, self).computeSize()
		
		return tuple([self.neurons])
	
	def compile(self):
		super(FC, self).compile()
		
		self.weights.weights = self.initializer['weights'].get(shape=(sum(self.in_size_flatten), self.neurons)) #np.random.rand(sum(self.in_size_flatten), self.neurons)
		self.weights.bias = self.initializer['bias'].get(shape=(1, self.neurons)) #np.random.rand(1, self.neurons)

	def forward(self, inputs):
		super(FC, self).forward(inputs)
		
		input = np.reshape(np.concatenate([i.flatten() for i in inputs], axis=-1), [-1, sum(self.in_size_flatten)])
		self.values.input = input
		out = np.dot(input, self.weights.weights) + self.weights.bias
		return np.reshape(out, [-1] + list(self.out_size))

	def derivatives(self, doutput):
		# BACKWARD
		# como la capa envia distintas derivadas a cada entrada, esta debe separar los pesos
		# Calculamos la backward con todos los pesos: (batch)x(neurons) [X] (neurons)x(I1 + I2 + ... In) = (batch)x(I1 + I2 + ... In)
		partial = np.transpose(self.weights.weights)
		global_backward = np.dot(doutput, partial)
		# las ponemos en formato flatten
		# se separa en cada input
		# [(batch)x(I1), (batch)xI2, ..., (batch)x(In)]
		backwards = np.split(global_backward, np.cumsum(self.in_size_flatten[:-1]), axis=-1)

		# WEIGHTS
		# para corregir los pesos estos se derivan con respecto los pesos
		partial_respect_w = np.transpose(self.values.input)
		# el resultado es una matriz de (input_size)x(output_size)
		w = np.dot(partial_respect_w, doutput)

		return backwards, (w, np.sum(doutput, axis=0))
