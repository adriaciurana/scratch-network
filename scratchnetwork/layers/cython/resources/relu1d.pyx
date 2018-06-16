import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
def nb_forward(np.ndarray[FLOAT64, ndim=2] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] out = np.zeros(shape=[batch_size, out_size0], dtype=np.float64)

	cdef unsigned int b, i
	for b in range(batch_size):
		for i in range(out_size0):
			if inputv[b, i] > 0:
				out[b, i] = inputv[b, i]
			else:
				out[b, i] = 0
	return out


# derivatives
def nb_derivatives(np.ndarray[FLOAT64, ndim=2] doutput, np.ndarray[FLOAT64, ndim=2] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] dx = np.zeros(shape=[batch_size, out_size0], dtype=np.float64)

	cdef unsigned int b, i
	for b in range(batch_size):
		for i in range(out_size0):
			if inputv[b, i] > 0:
				dx[b, i] = doutput[b, i]
			else:
				dx[b, i] = 0
	return dx