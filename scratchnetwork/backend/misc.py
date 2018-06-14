import numpy as np
class Misc:
	@staticmethod
	def add_pad(img, pad=(0,0)):
		return np.pad(img, [(0,0), (pad[0], pad[0]), (pad[1], pad[1]), (0,0)], 'constant')

	@staticmethod
	def inv_pad(img, pad=(0,0)):
		_, W, H, _ = img.shape
		return img[:, pad[0]:(W-pad[0]), pad[1]:(H-pad[1]), :]

	"""
		https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py
	"""
	@staticmethod
	def get_im2col_indices(x_shape, kernel_size, padding, stride):
		# First figure out what the size of the output should be
		N, H, W, C = x_shape
		assert (H + 2 * padding[0] - kernel_size[0]) % stride[0] == 0
		assert (W + 2 * padding[1] - kernel_size[0]) % stride[1] == 0
		out_height = (H + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
		out_width = (W + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

		i0 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
		i0 = np.tile(i0, C)
		i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
		j0 = np.tile(np.arange(kernel_size[1]), kernel_size[0] * C)
		j1 = stride[1] * np.tile(np.arange(out_width), out_height)
		i = i0.reshape(-1, 1) + i1.reshape(1, -1)
		j = j0.reshape(-1, 1) + j1.reshape(1, -1)

		k = np.repeat(np.arange(C), kernel_size[0] * kernel_size[1]).reshape(-1, 1)

		return (i, j, k)

	@staticmethod
	def im2col(img, kernel_size, padding=(0, 0), stride=(1, 1)):
		""" 
			An implementation of im2col based on some fancy indexing 
		"""
		# Zero-pad the input
		img_padded = np.pad(img, [(0,0), (padding[0], padding[0]), (padding[1], padding[1]), (0,0)], 'constant')

		i, j, k = Misc.get_im2col_indices(img.shape, kernel_size, padding=padding, stride=stride)
		cols = img_padded[:, i, j, k]
		# Batch, strides, data (kw, kh, num_filters)
		cols = cols.transpose(0, 2, 1)
		cols = cols.reshape(np.multiply.reduceat(cols.shape, (0, 2)))
		return cols, (i, j, k)

	@staticmethod
	def col2im(cols, x_shape, ijk, kernel_size, padding=(0,0)):
		"""
		1, 2, 0 -> 2, 0, 1
		0, 2, 1 -> 1, 0, 2
		"""
		i, j, k = ijk
		""" An implementation of col2im based on fancy indexing and np.add.at """
		N, H, W, C = x_shape
		H_padded, W_padded = H + 2 * padding[0], W + 2 * padding[1]
		x_padded = np.zeros((N, H_padded, W_padded, C), dtype=cols.dtype)
		cols_reshaped = cols.reshape(C * kernel_size[0] * kernel_size[1], -1, N)
		
		cols_reshaped = cols_reshaped.transpose(2, 0, 1)
		np.add.at(x_padded, (slice(None), i, j, k), cols_reshaped)
		return x_padded[:, padding[0]:(H - padding[0]), padding[1]:(W - padding[1]), :]

	"""
	@staticmethod
	def im2col2(img, kernel_size=(3,3), stride=(1,1)):
		#https://stackoverflow.com/questions/50292750/python-the-implementation-of-im2col-which-takes-the-advantages-of-6-dimensional
		N, H, W, C = img.shape
		NN, HH, WW, CC = img.strides
		out_h = (H - kernel_size[0])//stride[0] + 1
		out_w = (W - kernel_size[1])//stride[0] + 1
		col = np.lib.stride_tricks.as_strided(img, (N, out_h, out_w, C, kernel_size[0], kernel_size[1]), (NN, stride[0] * HH, stride[1] * WW, CC, HH, WW)).astype(float)
		return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))

	"""
