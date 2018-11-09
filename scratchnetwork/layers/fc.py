from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
class FC(Layer):
	def __init__(self, node, neurons, initializer={'weights': Initializer("lecun", "uniform"), 'bias': Initializer("zeros")}, params=None):
		if params is None:
			params = {}
		if 'weights_names' not in params:
			params['weights_names'] = ('weights', 'bias')
		params['number_of_inputs'] = 1
		super(FC, self).__init__(node, params=params)
		
		self.initializer = initializer
		self.neurons = neurons
	
	def computeSize(self):
		super(FC, self).computeSize()
		
		return tuple([self.neurons])
	
	def compile(self):
		super(FC, self).compile()
		self.weights.weights = self.initializer['weights'].get(shape=(sum(self.in_size_flatten), self.neurons))
		self.weights.bias = self.initializer['bias'].get(shape=[self.neurons])

	def forward(self, inputs):
		super(FC, self).forward(inputs)
			
		self.values.input = inputs[0]
		out = self.values.input.dot(self.weights.weights) + self.weights.bias
		return out.reshape([-1] + self.out_size)

	def derivatives(self, doutput):	
		# BACKWARD
		# como la capa envia distintas derivadas a cada entrada, esta debe separar los pesos
		# Calculamos la backward con todos los pesos: (batch)x(neurons) [X] (neurons)x(I1 + I2 + ... In) = (batch)x(I1 + I2 + ... In)
		dx = doutput.dot(self.weights.weights.T)
		# las ponemos en formato flatten
		# se separa en cada input
		# [(batch)x(I1), (batch)xI2, ..., (batch)x(In)]
		#backwards = np.split(global_backward, np.cumsum(self.in_size_flatten[:-1]), axis=-1)

		# WEIGHTS
		# para corregir los pesos estos se derivan con respecto los pesos
		# el resultado es una matriz de (input_size)x(output_size)
		dw =  self.values.input.T.dot(doutput)
		return dx, (dw, np.sum(doutput, axis=0))
	
	def save(self, h5_container, get_weights_id):
		layer_json = super(FC, self).save(h5_container, get_weights_id)
		layer_json['attributes']['neurons'] = self.neurons
		return layer_json

	def load(self, data, h5_container):
		super(FC, self).load(data, h5_container)
		self.neurons = data['attributes']['neurons']
