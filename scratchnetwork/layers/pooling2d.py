from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
from .cython import pooling2d

class Pooling2D(Layer):
	def __init__(self, node, type_pooling='max', pool_size=(2,2), stride=1, padding='valid', params={}):
		self.type_pooling = type_pooling
		self.pool_size = pool_size
		if isinstance(stride, (list, tuple)):
			self.stride = stride
		else:
			self.stride = (stride, stride)
		self.padding = padding

		if self.padding == 'valid':
			self.padding_size = (0, 0)

		elif self.padding == 'same':
			self.padding_size = (self.pool_size[0] // 2, self.pool_size[1] // 2)
		super(Pooling2D, self).__init__(node)

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
		elif self.type_pooling == 'mean':
			out = pooling2d.nb_forward_mean(input, self.pool_size, self.stride)


		"""Bind, Dind = np.meshgrid(range(out.shape[0]), range(out.shape[-1]))
		Bind = Bind.flatten()
		Dind = Dind.flatten()
		for i in range(self.out_size[0]):
			for j in range(self.out_size[1]):
				iin = i*self.stride[0]
				jin = j*self.stride[1]

				blockInput = input[:, iin:(iin + self.pool_size[0]), jin:(jin + self.pool_size[1]), :].transpose(0, 3, 1, 2).reshape([-1, np.prod(self.pool_size)])
				maxIndex = np.argmax(blockInput, axis=-1)
				mi, mj = np.unravel_index(maxIndex, self.pool_size)
				self.values.mask[Bind, iin + mi, jin + mj, Dind] = [i + 1, j + 1]
				out[:, i, j] = blockInput[range(maxIndex.size), maxIndex].reshape(out.shape[0], self.num_dim)
		"""
		return out

	def derivatives(self, doutput):
		if self.type_pooling == 'max':
			dx = pooling2d.nb_derivatives_max(doutput, self.in_size[0], self.values.mask, self.stride)
		elif self.type_pooling == 'mean':
			dx = pooling2d.nb_derivatives_mean(doutput, self.in_size[0], self.pool_size, self.stride)


		"""
		n = np.tile(np.reshape(range(doutput.shape[0]), [-1, 1, 1, 1]), [1] + list(self.in_size[0]))
		m = np.tile(np.reshape(range(self.num_dim), [1, 1, 1, -1]), [doutput.shape[0], self.in_size[0][1], self.in_size[0][2], 1])
		doutput = np.pad(doutput, [(0, 0), (1, 0), (1, 0), (0, 0)], mode='constant')
		dx = doutput[n, self.values.mask[:, :, :, :, 0], self.values.mask[:, :, :, :, 0], m]
		return dx
		"""
		return dx[:, self.padding_size[0]:(self.in_size[0][0] - self.padding_size[0]), self.padding_size[1]:(self.in_size[0][1] - self.padding_size[1]), :]