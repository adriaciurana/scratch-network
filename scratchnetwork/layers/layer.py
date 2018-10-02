import h5py
import copy
import sys
import numpy as np
from ..backend.exceptions import Exceptions
from ..backend.misc import Misc
from ..regularizators.regularizator import Regularizator

#import time
#t = time.time()
class Layer(object):
	LAYER_COUNTER = 0

	def __init__(self, node, func_repr_weights=lambda x: x, params=None):
		Layer.LAYER_COUNTER += 1
		self.LAYER_COUNTER = Layer.LAYER_COUNTER
		if params is None:
			params = {}

		self.node = node
		# pesos de la capa
		self.weights = Layer.Weights(func_repr_weights)
		if 'weights_names' in params:
			self.weights_names = params['weights_names']
			del params['weights_names']
		else:
			self.weights_names = None
		# valores intermedios de la capa
		self.values = Layer.Values()
		# parametros de la capa
		self.params = params

		# regularizacion
		self.regularizator = None
		self.is_trainable = True
		self.compute_backward = True

	def computeSize(self):
		if 'same_input_shape' in self.params:
			if self.in_size[:1] == self.in_size[:-1]:
				raise Exceptions.DifferentInputShape("El tipo de datos en la capa " \
					 + self.node.name + " debe ser igual en todos los casos. "+str(self.in_size))

	def compile(self):
		# inicializaciones
		if 'compute_forward_in_prediction' in self.params:
			self.node.compute_forward_in_prediction = self.params['compute_forward_in_prediction']
			del self.params['compute_forward_in_prediction']

		if 'compute_backward' in self.params:
			self.compute_backward = self.params['compute_backward']
			del self.params['compute_backward']

		if 'number_of_inputs' in self.params:
			if self.params['number_of_inputs'] < len(self.node.prevs):
				raise Exceptions.NumberInputsException("Numero de entradas excedidas (" \
				 + str(self.params['number_of_inputs']) + ") en " + type(self).__name__ + ":" + self.node.name)
			
			elif self.params['number_of_inputs'] > len(self.node.prevs):
				raise Exceptions.NumberInputsException("Numero de entradas inferiores (" \
				 + str(self.params['number_of_inputs']) + ") en " + type(self).__name__ + ":" + self.node.name)
			del self.params['number_of_inputs']
		
		if 'regularizator' in self.params:
			self.regularizator = self.params['regularizator']
			del self.params['regularizator']
		
	def forward(self, inputs):
		pass
	
	def derivatives(self, doutput=None):
		raise NotImplemented
	"""
		WEIGHTS:
	"""
	class Weights(object):
		def __init__(self, func_repr=lambda x: x):
			self.func_repr = func_repr
		def copy(self):
			c = self.__class__
			copy_weights_instance = c.__new__(c)
			
			for w in self.__dict__:
				if w == 'func_repr':
					continue
				setattr(copy_weights_instance, w, copy.copy(getattr(self, w)))
			return copy_weights_instance

		def get(self, name):
			if isinstance(self.func_repr, dict):
				return self.func_repr[name](getattr(self, name))
			return self.func_repr(getattr(self, name))

		def save(self, h5_container):
			for w in self.__dict__:
				if w == 'func_repr':
					continue
				attr = getattr(self, w)
				h5_container.create_dataset(w, data=attr, dtype=attr.dtype)

		def load(self, h5_container):
			for name in h5_container.keys():
				attr = h5_container[name].value
				setattr(self, name, attr)

	class Values(object):
		def copy(self):
			c = self.__class__
			copy_values_instance = c.__new__(c)
			
			for v in self.__dict__: # __attrs__
				setattr(copy_values_instance, v, copy.copy(getattr(self, v)))
			return copy_values_instance

	def getRegularization(self, name, weight):
		if isinstance(self.regularizator, dict):
			return self.regularizator[name].function(weight)
		else:
			if self.regularizator is None:
				return 0
			return self.regularizator.function(weight)
	""" 
	Para actualizar un peso se necesitan diversos parametros:
		Loss: donde se esta contribuyendo, desconocemos el funcional que se ha usado 
			para unir los batches (Normalmente es la media). Pero es importante saber que cada loss es acumulada de forma independiente.
		weights_losses: Indica el peso de cada loss, por defecto este es 1/num_losses
		name: indica el nombre del parametro a actualizar
	"""
	def correctWeight(self, label, name, dweight):
		# primero obtenemos el peso a corregir
		w = getattr(self.weights, name)
		# aplicamos el funcional respecto al batch
		dweight /= self.node.network.batch_size
		# añadimos la regularizacion
		# actualizamos la derivada
		dweight = dweight + self.getRegularization(name, w)

		# realizamos la correccion con respecto al optimizador
		iweight = self.node.network.optimizer.step(label, name, dweight)
		setattr(self.weights, name, w + iweight)

	def correctWeights(self, dweights):
		#import time
		#a0 = time.time()
		if isinstance(dweights, (list, tuple)):
			for i, dw in enumerate(dweights):
				# aplicamos las correciones a los pesos
				self.correctWeight(self.node.label, self.weights_names[i], dw)
		else:
			self.correctWeight(self.node.label, self.weights_names[0], dweights)

	"""
		COPY
	"""
	def copy(self, node):
		c = self.__class__
		copy_layer_instance = c.__new__(c)
		copy_layer_instance.node = node
		copy_layer_instance.weights = self.weights.copy()
		copy_layer_instance.values = self.values.copy()
		Layer.LAYER_COUNTER += 1
		copy_layer_instance.LAYER_COUNTER = Layer.LAYER_COUNTER
		for attr in self.__dict__:
			if attr in ['weights', 'values', 'LAYER_COUNTER', 'node']:
				continue
			setattr(copy_layer_instance, attr, copy.copy(getattr(self, attr)))
		"""print(Layer.LAYER_COUNTER)
		print(self.__dict__)
		print('------')
		print(copy_layer_instance.__dict__)"""

		return copy_layer_instance

	"""
		Properties
	"""
	
	@property
	def predict_flag(self):
		return self.node.network.predict_flag

	@property
	def batch_size(self):
		return self.node.network.batch_size

	"""
		Save
	"""
	def save(self, h5_container):
		layer_json = {'type': self.__class__.__name__, 'module': self.__class__.__module__, 'attributes':{}}
		layer_json['hash'] = Misc.hash(layer_json['module'], layer_json['type'])
		if self.regularizator is not None:
			layer_json['attributes']['regularizator'] = self.regularizator.save(h5_container.create_group("regularizator"))
		self.weights.save(h5_container.create_group("weights"))
		layer_json['attributes']['weights_names'] = self.weights_names
		layer_json['attributes']['is_trainable'] = self.is_trainable
		layer_json['attributes']['compute_backward'] = self.compute_backward
		layer_json['attributes']['params'] = self.params
		layer_json['attributes']['sizes'] = {}
		layer_json['attributes']['sizes']['in_size'] = self.in_size
		layer_json['attributes']['sizes']['in_size_flatten'] = self.in_size_flatten
		layer_json['attributes']['sizes']['out_size'] = self.out_size
		layer_json['attributes']['sizes']['out_size_flatten'] = self.out_size_flatten
		
		return layer_json

	def load(self, data, h5_container):
		pass


	@staticmethod
	def load_static(node, data, h5_container):
		if not Misc.check_hash(data['module'], data['type'], data['hash']):
			raise IndexError # Error
		my_class = Misc.import_class(data['module'], data['type'])
		obj = my_class.__new__(my_class)
		if 'regularizator' in data['attributes']:
			obj.regularizator = Regularizator.load_static(data['attributes']['regularizator'], h5_container['regularizator'])
		else:
			obj.regularizator = None
		obj.weights = Layer.Weights()
		obj.weights.load(h5_container['weights'])
		obj.values = Layer.Values()
		obj.weights_names = data['attributes']['weights_names']
		obj.is_trainable = data['attributes']['is_trainable']
		obj.compute_backward = data['attributes']['compute_backward']
		obj.params = data['attributes']['params']
		obj.node = node
		obj.in_size = data['attributes']['sizes']['in_size']
		obj.in_size_flatten = data['attributes']['sizes']['in_size_flatten']
		obj.out_size = data['attributes']['sizes']['out_size']
		obj.out_size_flatten = data['attributes']['sizes']['out_size_flatten']
		obj.load(data, h5_container)
		return obj