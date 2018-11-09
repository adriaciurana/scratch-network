from .layer import Layer
from ..backend.exceptions import Exceptions
import numpy as np
class Operation(Layer):
	ADD, SUB, PROD, DIV = range(4)
	__LUT = {'+': ADD, '-': SUB, '*': PROD, '/': DIV, '*': PROD}

	def __init__(self, node, operation='+', params=None):
		if params is None:
			params = {}
		params['number_of_inputs'] = 2
		super(Operation, self).__init__(node, params=params)
		if operation not in self.__LUT:
			raise Exceptions.OperationNotExist("La operacion en la capa " \
					 + self.node.name + " no existe.")
		self.operation = self.__LUT[operation]
		self.eps = numpy.finfo(np.float64).eps
	
	def computeSize(self):
		super(Operation, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Operation, self).forward(inputs)
		a, b = inputs
		if self.operation == self.ADD:
			return a + b
		
		elif self.operation == self.SUB:
			return a - b
		
		elif self.operation == self.PROD:
			self.values.a = a
			self.values.b = b
			return a*b
		
		elif self.operation == self.DIV:
			self.values.a = a
			self.values.b = b
			return a/(b + self.eps)

	def derivatives(self, doutput):
		if self.operation == self.ADD:
			return (doutput, doutput), None
		
		elif self.operation == self.SUB:
			return (doutput, -doutput), None
		
		elif self.operation == self.PROD:
			return (self.values.b*doutput, self.values.a*doutput), None
		
		elif self.operation == self.DIV:
			return (doutput/(b + self.eps), -(self.values.a/(self.values.b**2 + self.eps))*doutput), None

	def save(self, h5_container, get_weights_id):
		layer_json = super(Reshape, self).save(h5_container, get_weights_id)
		layer_json['attributes']['shape'] = self.shape
		return layer_json
		
	def load(self, data, h5_container):
		super(Reshape, self).load(data, h5_container)
		self.shape = data['attributes']['shape']