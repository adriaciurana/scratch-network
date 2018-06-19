import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=2] inputv):
	cdef unsigned int size = inputv.shape[1], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] out = np.empty(shape=[batch_size, size], dtype=np.float64)

	cdef unsigned int b, i
	for b in range(batch_size):
		for i in range(size):
			if inputv[b, i] > 0:
				out[b, i] = inputv[b, i]
			else:
				out[b, i] = 0
	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=2] doutput, np.ndarray[FLOAT64, ndim=2] inputv):
	cdef unsigned int size = inputv.shape[1], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] dx = np.empty(shape=[batch_size, size], dtype=np.float64)

	cdef unsigned int b, i
	for b in range(batch_size):
		for i in range(size):
			if inputv[b, i] > 0:
				dx[b, i] = doutput[b, i]
			else:
				dx[b, i] = 0
	return dx