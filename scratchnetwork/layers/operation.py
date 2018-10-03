from .layer import Layer
from ..backend.exceptions import Exceptions
import numpy as np
class Operation(Layer):
	def __init__(self, node, operation='+', params=None):
		if params is None:
			params = {}
		params['number_of_inputs'] = 2
		super(Operation, self).__init__(node, params=params)
		if operation not in ['+', '-', 'x', '/']:
			raise Exceptions.OperationNotExist("La operacion en la capa " \
					 + self.node.name + " no existe.")
		self.operation = operation
		self.eps = numpy.finfo(np.float64).eps
	
	def computeSize(self):
		super(Operation, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(Operation, self).forward(inputs)
		a, b = inputs
		if self.operation == '+':
			return a + b
		elif self.operation == '-':
			return a - b
		elif self.operation == 'x':
			self.values.a = a
			self.values.b = b
			return a*b
		elif self.operation == '/':
			self.values.a = a
			self.values.b = b
			return a/(b + self.eps)

	def derivatives(self, doutput):
		if self.operation == '+':
			return (doutput, doutput), None
		elif self.operation == '-':
			return (doutput, -doutput), None
		elif self.operation == 'x':
			return (self.values.b*doutput, self.values.a*doutput), None
		elif self.operation == '/':
			return (doutput/(b + self.eps), -(self.values.a/(self.values.b**2 + self.eps))*doutput), None