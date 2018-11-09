from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
from .cython import pooling2d
from .cs231n import fast_layers

class Pooling2D(Layer):
	def __init__(self, node, type_pooling='max', pool_size=(2,2), stride=(1, 1), padding='valid', params=None):
		if params is None:
			params = {}

		self.type_pooling = type_pooling
		self.pool_size = tuple(pool_size)
		if isinstance(stride, (list, tuple)):
			self.stride = tuple(stride)
		else:
			self.stride = (stride, stride)
		self.padding = padding

		if self.padding == 'valid':
			self.padding_size = (0, 0)

		elif self.padding == 'same':
			self.padding_size = (self.pool_size[0] // 2, self.pool_size[1] // 2)
		super(Pooling2D, self).__init__(node, params=params)

	def computeSize(self):
		super(Pooling2D, self).computeSize()
		
		return (self.in_size[0][0] - self.pool_size[0] + 2*self.padding_size[0])//self.stride[0] + 1, \
		(self.in_size[0][1] - self.pool_size[1] + 2*self.padding_size[1])//self.stride[1] + 1, \
		self.in_size[0][-1]
	
	def compile(self):
		super(Pooling2D, self).compile()
		
		if len(self.in_size[0]) < 2:
			self.num_dim = 1
		else:
			self.num_dim = self.in_size[0][2]
	
	def forward(self, inputs):
		super(Pooling2D, self).forward(inputs)

		input = np.pad(inputs[0], [(0, 0), (self.padding_size[0], self.padding_size[0]), (self.padding_size[1], self.padding_size[1]), (0, 0)], mode='constant')
		if self.type_pooling == 'max':
			out, self.values.mask = pooling2d.nb_forward_max(input, self.pool_size, self.stride)
			#out, self.values.cache = fast_layers.max_pool_forward_fast(inputs[0].transpose(0, 3, 1, 2), {'pool_height': self.pool_size[0], 'pool_width': self.pool_size[1], 'stride': int(self.stride[0])})
			#out = out.transpose(0, 2, 3, 1)
		elif self.type_pooling == 'mean':
			out = pooling2d.nb_forward_mean(input, self.pool_size, self.stride)
		return out

	def derivatives(self, doutput):
		if self.type_pooling == 'max':
			dx = pooling2d.nb_derivatives_max(doutput, tuple(self.in_size[0]), self.values.mask, self.stride)
			#dx = fast_layers.max_pool_backward_fast(doutput.transpose(0, 3, 1, 2), self.values.cache)
			#dx = dx.transpose(0, 2, 3, 1)
		elif self.type_pooling == 'mean':
			dx = pooling2d.nb_derivatives_mean(doutput, tuple(self.in_size[0]), self.pool_size, self.stride)
		return dx[:, self.padding_size[0]:(self.in_size[0][0] - self.padding_size[0]), self.padding_size[1]:(self.in_size[0][1] - self.padding_size[1]), :]

	def save(self, h5_container, get_weights_id):
		layer_json = super(Pooling2D, self).save(h5_container, get_weights_id)
		layer_json['attributes']['type_pooling'] = self.type_pooling
		layer_json['attributes']['pool_size'] = self.pool_size
		layer_json['attributes']['stride'] = self.stride
		layer_json['attributes']['padding'] = self.padding
		layer_json['attributes']['padding_size'] = self.padding_size
		return layer_json
		
	def load(self, data, h5_container):
		super(Pooling2D, self).load(data, h5_container)
		self.type_pooling = data['attributes']['type_pooling']
		self.pool_size = tuple(data['attributes']['pool_size'])
		self.stride = tuple(data['attributes']['stride'])
		self.padding = data['attributes']['padding']
		self.padding_size = tuple(data['attributes']['padding_size'])