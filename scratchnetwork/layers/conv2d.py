from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
class Conv2D(Layer):
	def __init__(self, node, num_filters, kernel_size=(3,3), stride=1, padding='valid', initializer=Initializer("normal"), params={}):
		self.initializer = initializer
		self.num_filters = num_filters
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		if self.padding == 'valid':
			self.padding_size = 0

		def aux(x):
			x = np.reshape(x, [-1] + list(self.kernel_size) + [self.num_filters])
			return np.transpose(x, [0, 3, 1, 2])
		super(Conv2D, self).__init__(node, tuple(['kernels']), aux) #np.reshape(x, [-1, self.num_filters] + list(self.kernel_size)))
	
	def computeSize(self):
		super(Conv2D, self).computeSize()
		
		return ((self.in_size[0][0] - self.kernel_size[0] + 2*self.padding_size)//self.stride + 1, (self.in_size[0][1] - self.kernel_size[1] + 2*self.padding_size)//self.stride + 1, self.num_filters)
	
	def compile(self):
		super(Conv2D, self).compile()
		
		if len(self.in_size[0]) < 2:
			self.num_dim = 1
		else:
			self.num_dim = self.in_size[0][2]
		self.weights.kernels = self.initializer.get(shape=(self.kernel_size[0]*self.kernel_size[1]*self.num_dim, self.num_filters)) #np.random.rand(self.kernel_size[0]*self.kernel_size[1]*self.num_dim, self.num_filters)

	def forward(self, inputs):
		super(Conv2D, self).forward(inputs)

		self.values.input = inputs[0]
		out = np.zeros(shape=[self.values.input.shape[0]] + list(self.out_size))
		for i in range(0, self.out_size[0], self.stride):
			for j in range(0, self.out_size[1], self.stride):
				iin = self.stride*i
				jin = self.stride*j
				#print(np.reshape(self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :], [-1] + [self.weights.kernels.shape[0]]).shape, self.weights.kernels.shape)
				out[:, i, j] = np.dot(np.reshape(self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :], [-1] + [self.weights.kernels.shape[0]]), self.weights.kernels)
		return out

	def derivatives(self, doutput):
		dw = np.zeros(shape=(self.kernel_size[0]*self.kernel_size[1]*self.num_dim, self.num_filters))
		dx = np.zeros(shape=[self.values.input.shape[0]] + list(self.in_size[0]))
		for i in range(self.out_size[0]):
			for j in range(self.out_size[1]):
				iin = i*self.stride
				jin = j*self.stride
				#print(iin, jin, i, j)
				#print('d1>', self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :].shape)
				#print('d2>',doutput[:, i, j, :].shape)
				

				# weights
				# [WxHxID, 1] x [1, 1x1xOD]
				# producto cartesiano = [WxHxID, OD]
				#print(np.reshape(self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :], [-1] + [self.weights.kernels.shape[0]]).shape, \
				#	np.reshape(doutput[:, i, j, :], [-1, self.num_filters]).shape)
				dw += np.dot(np.transpose(np.reshape(self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :], [-1] + [self.weights.kernels.shape[0]])), np.reshape(doutput[:, i, j, :], [-1, self.num_filters]))

				# backward
				# [1, 1x1xOD] x [OD, WxHxID]
				# [1, WxHxID]
				#print('=>', np.reshape(doutput[:, i, j, :], [-1, self.num_filters]).shape, np.transpose(self.weights.kernels).shape)
				#print(np.dot(np.reshape(doutput[:, i, j, :], [-1, self.num_filters]), np.transpose(self.weights.kernels)).shape)
				#print('->', np.reshape(np.dot(np.reshape(doutput[:, i, j, :], [-1, self.num_filters]), np.transpose(self.weights.kernels)), [-1] + list(self.kernel_size)  + [self.num_dim] ).shape)
				dx[:, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :] += np.reshape(np.dot(np.reshape(doutput[:, i, j, :], [-1, self.num_filters]), np.transpose(self.weights.kernels)), [-1] + list(self.kernel_size) + [self.num_dim])
		return dx, dw
