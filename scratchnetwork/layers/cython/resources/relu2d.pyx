import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
def nb_forward(np.ndarray[FLOAT64, ndim=3] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], out_size1 = inputv.shape[2], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=3] out = np.zeros(shape=[batch_size, out_size0, out_size1], dtype=np.float64)

	cdef unsigned int b, i, j
	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				if inputv[b, i, j] > 0:
					out[b, i, j] = inputv[b, i, j]
				else:
					out[b, i, j] = 0
	return out


# derivatives
def nb_derivatives(np.ndarray[FLOAT64, ndim=3] doutput, np.ndarray[FLOAT64, ndim=3] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], out_size1 = inputv.shape[2], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=3] dx = np.zeros(shape=[batch_size, out_size0, out_size1], dtype=np.float64)

	cdef unsigned int b, i, j
	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				if inputv[b, i, j] > 0:
					dx[b, i, j] = doutput[b, i, j]
				else:
					dx[b, i, j] = 0
	return dx