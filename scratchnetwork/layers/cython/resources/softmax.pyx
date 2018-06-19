import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp 

from numpy.math cimport INFINITY
from types cimport FLOAT64

# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=2] inputv):
	cdef unsigned int out_size = inputv.shape[1], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] out = np.empty(shape=[batch_size, out_size], dtype=np.float64)
	
	cdef unsigned int b, i
	cdef double maxv = - INFINITY, den
	for b in range(batch_size):
		maxv = - INFINITY
		
		for i in range(out_size):
			if maxv < inputv[b, i]:
				maxv = inputv[b, i]
		den = 0.
		for i in range(out_size):
			out[b, i] = exp(inputv[b, i] - maxv)
			den += out[b, i]
		
		for i in range(out_size):
			out[b, i] /= den

	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=2] doutput, np.ndarray[FLOAT64, ndim=2] ovalue):
	cdef unsigned int out_size = ovalue.shape[1], \
	batch_size = ovalue.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] dx = np.empty(shape=[batch_size, out_size], dtype=np.float64)

	cdef unsigned int b, i, j
	for b in range(batch_size):
		for i in range(out_size):
			dx[b, i] = 0.
			for j in range(out_size):
				if i == j:
					dx[b, i] += ovalue[b, i]*(1 - ovalue[b, i])*doutput[b, j]
				else:
					dx[b, i] += - ovalue[b, i]*ovalue[b, j]*doutput[b, j]
	return dx
	"""
	cdef unsigned int b, i
	cdef double acc
	for b in range(batch_size):
		for i in range(out_size):
			dx[b, i] = doutput[b, i]*expIn[b, i]*(den[b] - expIn[b, i])/(den[b]*den[b])

	return dx
	"""