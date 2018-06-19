import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=4] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], out_size1 = inputv.shape[2], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=4] out = np.empty(shape=[batch_size, out_size0, out_size1, num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, n
	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				for n in range(num_dim):
					if inputv[b, i, j, n] > 0:
						out[b, i, j, n] = inputv[b, i, j, n]
					else:
						out[b, i, j, n] = 0
	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=4] doutput, np.ndarray[FLOAT64, ndim=4] inputv):
	cdef unsigned int out_size0 = inputv.shape[1], out_size1 = inputv.shape[2], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=4] dx = np.empty(shape=[batch_size, out_size0, out_size1, num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, n
	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				for n in range(num_dim):
					if inputv[b, i, j, n] > 0:
						dx[b, i, j, n] = doutput[b, i, j, n]
					else:
						dx[b, i, j, n] = 0
	return dx