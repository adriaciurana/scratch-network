import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=1] inputv):
	cdef unsigned long int size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=1] out = np.empty(shape=[size], dtype=np.float64)

	cdef unsigned long int i
	for i in range(size):
		if inputv[i] > 0:
			out[i] = inputv[i]
		else:
			out[i] = 0
	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=1] doutput, np.ndarray[FLOAT64, ndim=1] inputv):
	cdef unsigned long int size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=1] dx = np.empty(shape=[size], dtype=np.float64)

	cdef unsigned long int i
	for i in range(size):
		if inputv[i] > 0:
			dx[i] = doutput[i]
		else:
			dx[i] = 0
	return dx