from .layer import Layer
import numpy as np
from ..backend.exceptions import Exceptions
class Input(Layer):
	def __init__(self, node, shape, params={}):
		params['compute_backward'] = False
		
		super(Input, self).__init__(node, params)
		self.shape = tuple(shape)
		self.data = None
		self.has_data = False
		self.old_shape = None

	def fill(self, data):
		if data.shape[1:] != self.shape:
			raise Exceptions.InputShapeException("Los datos en la capa Input: " + self.node.name + " tiene un tamaño incorrecto. \
				Son "+str(data.shape[1:])+" y deberian ser " + str(self.shape) + ".")
		if self.data is not None:
			self.old_shape = self.data.shape
		self.data = data
		self.has_data = True
		self.batch_size = self.data.shape[0]

		# Se deben volver a inicializar los tests de tamaño
		if self.data.shape != self.old_shape:
			self.node.network.firstForward = True

	def computeSize(self):
		super(Input, self).computeSize()
		return self.shape

	def firstForward(self, inputs):
		super(Input, self).firstForward(inputs)
		if self.data is None:
			raise Exceptions.InputNotFillException("La capa Input: " + self.node.name + " no ha sido llenada.")

	def forward(self, inputs):
		return self.data

	def batchSize(self):
		return self.batch_size
