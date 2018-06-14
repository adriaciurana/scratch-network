import numpy as np
class Misc:
	@staticmethod
	def add_pad(img, pad=(0,0)):
		return np.pad(img, [(0,0), (pad[0], pad[0]), (pad[1], pad[1]), (0,0)], 'constant')

	@staticmethod
	def inv_pad(img, pad=(0,0)):
		_, W, H, _ = img.shape
		return img[:, pad[0]:(W-pad[0]), pad[1]:(H-pad[1]), :]

	@staticmethod
	def im2col(img, kernel_size=(3,3), stride=(1,1)):
		#https://stackoverflow.com/questions/50292750/python-the-implementation-of-im2col-which-takes-the-advantages-of-6-dimensional
		N, H, W, C = img.shape
		NN, HH, WW, CC = img.strides
		out_h = (H - kernel_size[0])//stride[0] + 1
		out_w = (W - kernel_size[1])//stride[0] + 1
		col = np.lib.stride_tricks.as_strided(img, (N, out_h, out_w, C, kernel_size[0], kernel_size[1]), (NN, stride[0] * HH, stride[1] * WW, CC, HH, WW)).astype(float)
		return col.reshape(np.multiply.reduceat(col.shape, (0, 3)))
