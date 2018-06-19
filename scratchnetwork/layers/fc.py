from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
from .cython import fc
import threading
class FC(Layer):
	def __init__(self, node, neurons, initializer={'weights': Initializer("normal"), 'bias': Initializer("normal")}, params={}):
		params['number_of_inputs'] = 1
		super(FC, self).__init__(node, weights_names=('weights', 'bias'), params=params)
		
		self.initializer = initializer
		self.neurons = neurons
	
	def computeSize(self):
		super(FC, self).computeSize()
		
		return tuple([self.neurons])
	
	def compile(self):
		super(FC, self).compile()
		
		self.weights.weights = self.initializer['weights'].get(shape=(sum(self.in_size_flatten), self.neurons))/sum(self.in_size_flatten) #np.random.rand(sum(self.in_size_flatten), self.neurons)
		self.weights.bias = self.initializer['bias'].get(shape=[self.neurons]) #np.random.rand(1, self.neurons)

	def forward(self, inputs):
		super(FC, self).forward(inputs)
		
		
		self.values.input = inputs[0]
		out = self.values.input.dot(self.weights.weights) + self.weights.bias
		return out.reshape([-1] + list(self.out_size))

		"""self.values.input = inputs[0]
		out = np.empty(shape=[inputs[0].shape[0]] + list(self.out_size))
		def thread_main(cidx):
			out[cidx:cidx+1] = fc.nb_forward(self.values.input[cidx:cidx+1], self.weights.weights, self.weights.bias)[0]
		threads = [threading.Thread(target=thread_main, args=(cidx,)) for cidx in range(self.values.input.shape[0])]
		
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()"""
		#out = fc.nb_forward(self.values.input, self.weights.weights, self.weights.bias)
		return out

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
		#dx, dw, db = fc.nb_derivatives(doutput, self.values.input, self.weights.weights)
		"""dx = np.empty(shape=[doutput.shape[0]] + [self.in_size_flatten[0]])
		dw = np.empty(shape=[doutput.shape[0]] + list(self.weights.weights.shape))
		db = np.empty(shape=[doutput.shape[0]] + list(self.weights.bias.shape))
		def thread_main(cidx):
			aux = fc.nb_derivatives(doutput[cidx:cidx+1], self.values.input[cidx:cidx+1], self.weights.weights)
			dx[cidx] = aux[0][0]
			dw[cidx] = aux[1]
			db[cidx] = aux[2]
		threads = [threading.Thread(target=thread_main, args=(cidx,)) for cidx in range(doutput.shape[0])]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		dw = dw.sum(axis=0)
		db = db.sum(axis=0)
		return np.reshape(dx, [-1] + list(self.in_size[0])), (dw, db)"""
