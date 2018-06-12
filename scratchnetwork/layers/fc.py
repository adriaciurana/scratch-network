from .layer import Layer
import numpy as np
class FC(Layer):
	def __init__(self, node, neurons, params={}):
		super(FC, self).__init__(node)
		
		self.neurons = neurons
	
	def computeSize(self):
		super(FC, self).computeSize()
		
		return tuple([self.neurons])
	
	def compile(self):
		super(FC, self).compile()
		
		self.weights.weights = np.random.rand(sum(self.in_size_flatten), self.neurons)
		self.weights.bias = np.random.rand(1, self.neurons)

	def forward(self, inputs):
		super(FC, self).forward(inputs)
		
		input = np.reshape(np.concatenate([i.flatten() for i in inputs], axis=-1), [-1, sum(self.in_size_flatten)])
		self.values.input = input
		out = np.dot(input, self.weights.weights) + self.weights.bias
		return np.reshape(out, [-1] + list(self.out_size))

	def backward(self, doutput):
		# como la capa envia distintas derivadas a cada entrada, esta debe separar los pesos
		# Calculamos la backward con todos los pesos: (batch)x(neurons) [X] (neurons)x(I1 + I2 + ... In) = (batch)x(I1 + I2 + ... In)
		partial = np.transpose(self.weights.weights)
		global_backward = np.dot(doutput, partial)
		# las ponemos en formato flatten
		# se separa en cada input
		# [(batch)x(I1), (batch)xI2, ..., (batch)x(In)]
		backwards = np.split(global_backward, np.cumsum(self.in_size_flatten[:-1]), axis=-1)
		return backwards

	def correctWeights(self, doutput):
		# para corregir los pesos estos se derivan con respecto los pesos
		partial_respect_w = np.transpose(self.values.input)
		# el resultado es una matriz de (input_size)x(output_size)
		w = np.dot(partial_respect_w, doutput)

		# aplicamos las correciones a los pesos
		self.correctWeight('weights', w)
		self.correctWeight('bias', np.mean(doutput, axis=0))
