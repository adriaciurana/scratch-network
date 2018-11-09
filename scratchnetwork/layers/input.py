from .layer import Layer
import numpy as np
from ..backend.exceptions import Exceptions
class Input(Layer):
	def __init__(self, node, shape, params=None):
		if params is None:
			params = {}
		params['compute_backward'] = False
		
		super(Input, self).__init__(node, params=params)
		self.shape = tuple(shape)
		self.data = None

	def fill(self, data):
		if data.shape[1:] != self.shape:
			raise Exceptions.InputShapeException("Los datos en la capa "+self.__class__.__name__+": " + self.node.name + " tiene un tamaño incorrecto. \
				Son "+str(data.shape[1:])+" y deberian ser " + str(self.shape) + ".")
		self.data = np.float64(data)
		
		self.batch_size_input = self.data.shape[0]

	def computeSize(self):
		super(Input, self).computeSize()
		return self.shape

	def forward(self, inputs):	
		if self.data is None:
			raise Exceptions.InputNotFillException("La capa "+self.__class__.__name__+": " + self.node.name + " no ha sido llenada.")
		aux = self.data
		self.data = None
		return aux

	@property
	def has_data(self):
		#print(self.node.name)
		
		return self.data is not None

	@property
	def batch_size(self):
		return self.batch_size_input

	def save(self, h5_container, get_weights_id):
		layer_json = super(Input, self).save(h5_container, get_weights_id)
		layer_json['attributes']['shape'] = self.shape
		return layer_json
		
	def load(self, data, h5_container):
		super(Input, self).load(data, h5_container)
		self.shape = tuple(data['attributes']['shape'])
