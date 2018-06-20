import cython
cimport cython
import numpy as np
cimport numpy as np

from numpy.math cimport INFINITY
from types cimport FLOAT64, UINT
from cython.parallel import prange,parallel
cimport openmp
from libc.stdlib cimport abort, malloc, free

# max
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward_max(np.ndarray[FLOAT64, ndim=4] inputv, tuple pool_size, tuple stride):
	cdef unsigned int pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]

	cdef unsigned int out_size0 = (inputv.shape[1] - pool_size0) / stride0 + 1, \
	out_size1 = (inputv.shape[2] - pool_size1) / stride1 + 1

	cdef np.ndarray[UINT, ndim=5] mask = np.empty(shape=[batch_size, out_size0, out_size1, num_dim, 2], dtype=np.uint)
	cdef np.ndarray[FLOAT64, ndim=4] out = np.empty(shape=[batch_size, out_size0, out_size1, num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, m, kw, kh, n
	cdef unsigned int iin, jin
	cdef double blockInput
	cdef double maxv
	cdef long int maxi, maxj

	for b in prange(batch_size, nogil=True):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					maxv = - INFINITY
					maxi = -1
					maxj = -1
					for kw in range(pool_size0):
						for kh in range(pool_size1):
							blockInput = inputv[b, iin + kw, jin + kh, m]
							if maxi == -1 or maxv < blockInput:
								maxv = blockInput
								maxi = kw
								maxj = kh
					mask[b, i, j, m, 0] = maxi
					mask[b, i, j, m, 1] = maxj
					out[b, i, j, m] = maxv

	return out, mask

@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives_max(np.ndarray[FLOAT64, ndim=4] doutput, tuple input_size, np.ndarray[UINT, ndim=5] mask, tuple stride):
	cdef unsigned int out_size0 = doutput.shape[1], out_size1 = doutput.shape[2], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = doutput.shape[3], \
	batch_size = doutput.shape[0]


	cdef np.ndarray[FLOAT64, ndim=4] dx = np.zeros(shape=[batch_size, input_size[0], input_size[1], num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, m, n
	cdef unsigned int idx, iin, jin
	cdef double blockInput
	cdef unsigned int iin_kw, jin_kh

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					iin_kw = iin + mask[b, i, j, m, 0]
					jin_kh = jin + mask[b, i, j, m, 1]
					dx[b, iin_kw, jin_kh, m] = doutput[b, i, j, m]
	return dx

# mean
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward_mean(np.ndarray[FLOAT64, ndim=4] inputv, tuple pool_size, tuple stride):
	cdef unsigned int pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]

	cdef unsigned int out_size0 = (inputv.shape[1] - pool_size0) / stride0 + 1, \
	out_size1 = (inputv.shape[2] - pool_size1) / stride1 + 1

	cdef np.ndarray[FLOAT64, ndim=4] out = np.empty(shape=[batch_size, out_size0, out_size1, num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, m, kw, kh, n
	cdef unsigned int idx, iin, jin
	cdef double blockInput
	cdef double den = pool_size0*pool_size1

	for b in prange(batch_size, nogil=True):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					out[b, i, j, m] = 0.
					for kw in range(pool_size0):
						for kh in range(pool_size1):
							out[b, i, j, m] += inputv[b, iin + kw, jin + kh, m]
					out[b, i, j, m] /= den

	return out

@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives_mean(np.ndarray[FLOAT64, ndim=4] doutput, tuple input_size, tuple pool_size, tuple stride):
	cdef unsigned int pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	out_size0 = doutput.shape[1], out_size1 = doutput.shape[2], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = doutput.shape[3], \
	batch_size = doutput.shape[0]

	cdef np.ndarray[FLOAT64, ndim=4] dx = np.zeros(shape=[batch_size, input_size[0], input_size[1], num_dim], dtype=np.float64)

	cdef unsigned int b, i, j, m, kw, kh, n
	cdef unsigned int idx, iin, jin
	cdef double blockInput
	cdef unsigned int iin_kw, jin_kh
	
	cdef double den = pool_size0*pool_size1
	cdef double dx_p


	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					dx_p = doutput[b, i, j, m] / den
					for kw in range(pool_size0):
						for kh in range(pool_size1):
							iin_kw = iin + kw
							jin_kh = jin + kh
							dx[b, iin_kw, jin_kh, m] += dx_p
	return dx