from .layer import Layer
import numpy as np
from ..backend.exceptions import Exceptions
class Custom(Layer):
	def __init__(self, node, forward, backward, compute_size, compile, params=None):
		if params is None:
			params = {}
		params['number_of_inputs'] = 1
		super(Custom, self).__init__(node, params=params)
		self._func_forward = forward
		self._func_backward = backward
		self._func_compute_size = compute_size
		self._func_compile = compile
		
		self.constants = Layer.Values()
	
	def computeSize(self):
		super(Custom, self).computeSize()
		return self._func_compute_size(self.constants, self.in_size)

	def compile(self):
		super(Custom, self).compile()
		self._func_compile(self.constants, self.weights)

	def forward(self, inputs):
		super(Custom, self).forward(inputs, self.constants, self.weights, self.values)
		return self._func_forward(inputs)

	def derivatives(self, doutput):
		return self._func_backward(doutput, self.constants, self.weights, self.values)

	def __error(self):
		raise Exceptions.CustomNotCompile('La capa Custom no puede compilarse sin haberle pasado las funciones compute_size y compile')

	def set(self, forward, backward, compute_size=__error, compile=__error):
		self._func_forward = forward
		self._func_backward = backward
		self._func_compute_size = compute_size
		self._func_compile = compile

