import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64

# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=2] inputv, np.ndarray[FLOAT64, ndim=2] weights, np.ndarray[FLOAT64, ndim=1] bias):
	cdef unsigned long int batch_size = inputv.shape[0], \
	num_dims = weights.shape[0], \
	num_filters = weights.shape[1]

	cdef np.ndarray[FLOAT64, ndim=2] out = np.empty(shape=[batch_size, num_filters], dtype=np.float64)

	cdef unsigned long int b, m, n
	for b in range(batch_size):
		for n in range(num_filters):
			out[b, n] = bias[n]
			for m in range(num_dims):
				out[b, n] += inputv[b, m]*weights[m, n]
	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=2] doutput, np.ndarray[FLOAT64, ndim=2] inputv, np.ndarray[FLOAT64, ndim=2] weights):
	cdef unsigned long int batch_size = inputv.shape[0], \
	num_dims = weights.shape[0], \
	num_filters = weights.shape[1]

	cdef np.ndarray[FLOAT64, ndim=2] dx = np.zeros(shape=[batch_size, num_dims], dtype=np.float64)
	cdef np.ndarray[FLOAT64, ndim=2] dw = np.zeros(shape=[num_dims, num_filters], dtype=np.float64)
	cdef np.ndarray[FLOAT64, ndim=1] db = np.zeros(shape=[num_filters], dtype=np.float64)


	cdef unsigned long int b, m, n
	cdef double output_ptr
	for b in range(batch_size):
		for n in range(num_filters):
			output_ptr = doutput[b, n]
			db[n] += output_ptr
			for m in range(num_dims):
				dx[b, m] += output_ptr*weights[m, n]
				dw[m, n] += output_ptr*inputv[b, m]
	return dx, dw, db