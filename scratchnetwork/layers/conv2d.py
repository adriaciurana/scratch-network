from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
from .cython import conv2d
import threading

class Conv2D(Layer):
	def __init__(self, node, num_filters, kernel_size=(3,3), stride=(1, 1), padding='valid', initializer={'weights': Initializer("lecun"), 'bias': Initializer("normal")}, params=None):
		if params is None:
			params = {}

		self.initializer = initializer
		self.num_filters = num_filters
		self.kernel_size = tuple(kernel_size)
		if isinstance(stride, (list, tuple)):
			self.stride = tuple(stride)
		else:
			self.stride = (stride, stride)
		self.padding = padding

		if self.padding == 'valid':
			self.padding_size = (0, 0)

		elif self.padding == 'same':
			self.padding_size = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

		if 'weights_names' not in params:
			params['weights_names'] = ('kernels', 'bias')
		super(Conv2D, self).__init__(node, func_repr_weights=lambda x: \
			np.transpose(x, [0, 3, 1, 2]), params=params)
		
	def computeSize(self):
		super(Conv2D, self).computeSize()
		
		return (self.in_size[0][0] - self.kernel_size[0] + 2*self.padding_size[0])//self.stride[0] + 1, \
			(self.in_size[0][1] - self.kernel_size[1] + 2*self.padding_size[1])//self.stride[1] + 1, \
			self.num_filters
	
	def compile(self):
		super(Conv2D, self).compile()
		
		if len(self.in_size[0]) <= 2:
			self.num_dim = 1
		else:
			self.num_dim = self.in_size[0][2]
		self.weights.kernels = self.initializer['weights'].get(shape=(self.num_dim, self.kernel_size[0], self.kernel_size[1], self.num_filters)) / (self.kernel_size[0]*self.kernel_size[1]*self.num_filters)
		self.weights.bias = self.initializer['bias'].get(shape=[self.num_filters])

	def forward(self, inputs):
		super(Conv2D, self).forward(inputs)

		self.values.input = np.pad(inputs[0], [(0, 0), (self.padding_size[0], self.padding_size[0]), (self.padding_size[1], self.padding_size[1]), (0, 0)], mode='constant')
		out = conv2d.nb_forward(self.values.input, self.weights.kernels, self.weights.bias, self.stride)
		return out


	def derivatives(self, doutput):
		dx, dw, db = conv2d.nb_derivatives(doutput, self.values.input, self.weights.kernels, self.stride)		
		# devolvemos resultados
		return dx[:, self.padding_size[0]:(self.in_size[0][0] - self.padding_size[0]), self.padding_size[1]:(self.in_size[0][1] - self.padding_size[1]), :], (dw, db)

	def save(self, h5_container):
		layer_json = super(Conv2D, self).save(h5_container)
		layer_json['attributes']['num_filters'] = self.num_filters
		layer_json['attributes']['kernel_size'] = self.kernel_size
		layer_json['attributes']['stride'] = self.stride
		layer_json['attributes']['padding'] = self.padding
		layer_json['attributes']['padding_size'] = self.padding_size
		return layer_json
		
	def load(self, data, h5_container):
		super(Conv2D, self).load(data, h5_container)
		self.num_filters = data['attributes']['num_filters']
		self.kernel_size = tuple(data['attributes']['kernel_size'])
		self.stride = tuple(data['attributes']['stride'])
		self.padding = data['attributes']['padding']
		self.padding_size = tuple(data['attributes']['padding_size'])