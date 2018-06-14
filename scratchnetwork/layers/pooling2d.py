from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
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

		self.values.input = np.pad(inputs[0], [(0, 0), (self.padding_size[0], self.padding_size[0]), (self.padding_size[1], self.padding_size[1]), (0, 0)], mode='constant')
		out = np.zeros(shape=[self.values.input.shape[0]] + list(self.out_size))

		# Lo realizamos sin poner el if dentro del bucle para optimizar un poco el codigo
		if self.type_pooling == 'max':
			if not self.predict_flag:
				self.values.positions = np.zeros(shape=[self.values.input.shape[0]] + list(self.in_size[0]))
			
			for i in range(self.out_size[0]):
				for j in range(self.out_size[1]):
					iin = i*self.stride[0]
					jin = j*self.stride[1]

					blockInput = self.values.input[:, iin:(iin + self.pool_size[0]), jin:(jin + self.pool_size[1]), :]
					blockInputFlatten = np.reshape(blockInput, [-1, blockInput.shape[1]*blockInput.shape[2], blockInput.shape[3]])
					for m, pb in enumerate(np.argmax(blockInputFlatten, axis=1)):
						for n, pf in enumerate(pb):
							p = np.unravel_index(pf, blockInput.shape)
							out[m, i, j, n] = blockInput[p]
							if self.predict_flag:
								p = (p[0], (p[1] + i)*self.stride[0], (p[2] + j)*self.stride[1], p[3])
								self.values.positions[m, i, j, n] = 1 + np.ravel_multi_index(p, blockInput.shape)
		return out

	def derivatives(self, doutput):
		def map_doutput(x):
			if x == 0:
				return 0
			return doutput[x - 1]
		func = np.vectorize(map_doutput)
		
		if self.type_pooling == 'max':
			return func(self.values.positions)