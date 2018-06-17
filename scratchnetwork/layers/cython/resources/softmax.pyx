import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp 

from numpy.math cimport INFINITY
from types cimport FLOAT64

# forward
def nb_forward(np.ndarray[FLOAT64, ndim=2] ovalue):
	cdef unsigned int out_size0 = ovalue.shape[1], \
	batch_size = ovalue.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] out = np.zeros(shape=[batch_size, out_size0], dtype=np.float64)

	cdef unsigned int b, i
	cdef double maxv = - INFINITY, den
	for b in range(batch_size):
		maxv = - INFINITY
		for i in range(out_size0):
			if maxv < ovalue[b, i]:
				maxv = ovalue[b, i]
		den = 0.		
		for i in range(out_size0):
			out[b, i] = exp(ovalue[b, i] - maxv)
			den += out[b, i]
		
		for i in range(out_size0):
			out[b, i] /= den

	return out


# derivatives
def nb_derivatives(np.ndarray[FLOAT64, ndim=2] doutput, np.ndarray[FLOAT64, ndim=2] ovalue):
	cdef unsigned int out_size0 = ovalue.shape[1], \
	batch_size = ovalue.shape[0]

	cdef np.ndarray[FLOAT64, ndim=2] dx = np.zeros(shape=[batch_size, out_size0], dtype=np.float64)

	"""cdef unsigned int b, i
	def double acc, aux
	for b in range(batch_size):
		for i in range(out_size0):
			acc = 0.
			aux = ovalue[b, i]
			for j in range(out_size0):
				if i == j:
					acc += aux*(1 - aux)
				else:
					acc += - aux*ovalue[b, j]

			dx[b, i] = doutput[b, i]*acc
	"""
	cdef unsigned int b, i
	cdef double acc
	for b in range(batch_size):
		acc = 0.
		for i in range(out_size0):
				acc += ovalue[b, i]

		for i in range(out_size0):
			dx[b, i] = doutput[b, i]*ovalue[b, i]*(acc - ovalue[b, i])/(acc*acc)
	return dx