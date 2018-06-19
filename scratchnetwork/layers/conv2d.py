from .layer import Layer
from ..backend.initializer import Initializer
import numpy as np
from .cython import conv2d
import threading

class Conv2D(Layer):
	def __init__(self, node, num_filters, kernel_size=(3,3), stride=(1, 1), padding='valid', initializer={'weights': Initializer("lecun"), 'bias': Initializer("normal")}, params={}):
		self.initializer = initializer
		self.num_filters = num_filters
		self.kernel_size = kernel_size
		if isinstance(stride, (list, tuple)):
			self.stride = stride
		else:
			self.stride = (stride, stride)
		self.padding = padding

		if self.padding == 'valid':
			self.padding_size = (0, 0)

		elif self.padding == 'same':
			self.padding_size = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

		super(Conv2D, self).__init__(node, weights_names=('kernels', 'bias'), func_repr_weights=lambda x: \
			np.transpose(x, [0, 3, 1, 2]), params=params) #np.reshape(x, [-1, self.num_filters] + list(self.kernel_size)))

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
		self.weights.kernels = self.initializer['weights'].get(shape=(self.num_dim, self.kernel_size[0], self.kernel_size[1], self.num_filters))/(self.kernel_size[0] * self.kernel_size[1] * self.num_dim)
		self.weights.bias = self.initializer['bias'].get(shape=[self.num_filters])

	def forward(self, inputs):
		super(Conv2D, self).forward(inputs)

		self.values.input = np.pad(inputs[0], [(0, 0), (self.padding_size[0], self.padding_size[0]), (self.padding_size[1], self.padding_size[1]), (0, 0)], mode='constant')
		
		"""out = np.empty(shape=[inputs[0].shape[0]] + list(self.out_size))
		def thread_main(cidx):
			out[cidx:cidx+1] = conv2d.nb_forward(self.values.input[cidx:cidx+1], self.weights.kernels, self.weights.bias, self.stride)[0]
		threads = [threading.Thread(target=thread_main, args=(cidx,)) for cidx in range(self.values.input.shape[0])]
		
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		"""
		out = conv2d.nb_forward(self.values.input, self.weights.kernels, self.weights.bias, self.stride)
		"""
		for i in range(self.out_size[0]):
			for j in range(self.out_size[1]):
				iin = i*self.stride[0]
				jin = j*self.stride[1]
				
				blockInput = self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :].reshape([-1] + [self.weights.kernels.shape[0]])
				out[:, i, j] = blockInput.dot(self.weights.kernels) + self.weights.bias
		"""
		return out


	def derivatives(self, doutput):
		"""dx = np.empty(shape=[doutput.shape[0], self.in_size[0][0] + 2*self.padding_size[0], self.in_size[0][1] + 2*self.padding_size[1], self.in_size[0][2]])
		dw = np.empty(shape=[doutput.shape[0]] + list(self.weights.kernels.shape))
		db = np.empty(shape=[doutput.shape[0]] + list(self.weights.bias.shape))
		def thread_main(cidx):
			aux = conv2d.nb_derivatives(doutput[cidx:cidx+1], self.values.input[cidx:cidx+1], self.weights.kernels, self.stride)
			dx[cidx] = aux[0][0]
			dw[cidx] = aux[1]
			db[cidx] = aux[2]
		threads = [threading.Thread(target=thread_main, args=(cidx,)) for cidx in range(doutput.shape[0])]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		dw = dw.sum(axis=0)
		db = db.sum(axis=0)"""
		dx, dw, db = conv2d.nb_derivatives(doutput, self.values.input, self.weights.kernels, self.stride)
		#print([dw.max(), dw.min()], [db.max(), db.min()])
		"""
		for i in range(self.out_size[0]):
			for j in range(self.out_size[1]):
				iin = i*self.stride[0]
				jin = j*self.stride[1]
				
				# weights
				# [WxHxID, 1] x [1, 1x1xOD]
				# producto cartesiano = [WxHxID, OD]
				blockInput = self.values.input[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :].reshape([-1] + [self.weights.kernels.shape[0]]).T
				doutput_raw = doutput[:, i, j, :].reshape([-1, self.num_filters])
				dw += blockInput.dot(doutput_raw)
				db += np.sum(doutput_raw, axis=0)

				# backward
				# [1, 1x1xOD] x [OD, WxHxID]
				# [1, WxHxID]
				dx_p = doutput_raw.dot(self.weights.kernels.T).reshape([-1] + list(self.kernel_size) + [self.num_dim])
				dx[:, iin:(iin + self.kernel_size[0]), jin:(jin + self.kernel_size[1]), :] += dx_p
		"""
		
		# devolvemos resultados
		return dx[:, self.padding_size[0]:(self.in_size[0][0] - self.padding_size[0]), self.padding_size[1]:(self.in_size[0][1] - self.padding_size[1]), :], (dw, db)


	# version numpy no es mas rapida
	# def forward(self, inputs):
	# 	super(Conv2D, self).forward(inputs)

	# 	self.values.input, self.values.indexes = Misc.im2col(inputs[0], kernel_size=self.kernel_size, padding=self.padding_size, stride=self.stride)
	# 	input = np.dot(self.values.input, self.weights.kernels)
	# 	return np.reshape(input, [-1] + list(self.out_size))
	# 	"""
	# 	input = Misc.add_pad(inputs[0], pad=self.padding_size)

	# 	if not self.predict_flag:
	# 		self.values.input = Misc.im2col(input, kernel_size=self.kernel_size, stride=(1,1))
	# 		input = np.reshape(self.values.input, [-1, 
	# 					(self.in_size[0][0] - self.kernel_size[0] + 2*self.padding_size[0]) + 1, 
	# 					(self.in_size[0][1] - self.kernel_size[1] + 2*self.padding_size[1]) + 1, 
	# 					self.values.input.shape[1]])
	# 		input = np.reshape(input[:, ::self.stride[0], ::self.stride[1], :], [-1, self.values.input.shape[1]])
	# 		input = np.dot(input, self.weights.kernels)
	# 		return np.reshape(input, [-1] + list(self.out_size))
	# 	else:
	# 		input = Misc.im2col(input, kernel_size=self.kernel_size, stride=self.stride)
	# 		input = np.dot(input, self.weights.kernels)
	# 		return np.reshape(input, [-1] + list(self.out_size))
	# 	"""

	# def derivatives(self, doutput):
	# 	doutput = doutput.reshape([-1, self.num_filters])
	# 	dx = np.dot(doutput, self.weights.kernels.T)
	# 	dx = Misc.col2im(self.values.input, [doutput.shape[0]] + list(self.in_size[0]), self.values.indexes, kernel_size=self.kernel_size, padding=(0,0))
	# 	"""
	# 	dw = doutput.T.dot(self.values.input).T
	# 	return dx, dw"""
	# 	"""
	# 	#input = Misc.im2col(self.values.input, kernel_size=self.kernel_size, stride=(1,1))
	# 	doutput_reshape = np.reshape(doutput, [-1] + list(self.out_size))
	# 	dout_strided = np.zeros(shape=(doutput.shape[0], self.out_size[0] * self.stride[0], self.out_size[1] * self.stride[1], self.out_size[2]))
	# 	dout_strided[:, ::self.stride[0], ::self.stride[1], :] = doutput_reshape
	# 	dout_mat = np.reshape(dout_strided, [-1, self.num_filters])
	# 	dw = np.dot(np.transpose(self.values.input), dout_mat)
	# 	dw = np.reshape(np.dot(np.transpose(self.values.input), dout_mat), list(self.weights.kernels.shape))
		
	# 	dx = np.reshape(np.dot(dout_mat, np.transpose(self.weights.kernels)), 
	# 		[-1] + [np.prod(self.out_size)] + list(self.weights.kernels.shape))
	# 	dx = np.sum(dx, axis=1)
	# 	# falta aumentar stride
	# 	dx = np.reshape(dx, [-1] + list(self.in_size[0]))
	# 	dx = Misc.inv_pad(dx, pad=self.padding_size)
	# 	return dx, dw"""