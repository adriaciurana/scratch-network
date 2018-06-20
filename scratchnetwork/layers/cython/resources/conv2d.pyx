import cython
cimport cython
import numpy as np
cimport numpy as np

from types cimport FLOAT64
from cython.parallel import prange,parallel
cimport openmp
from libc.stdlib cimport abort, malloc, free


# forward
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_forward(np.ndarray[FLOAT64, ndim=4] inputv, np.ndarray[FLOAT64, ndim=4] kernels, np.ndarray[FLOAT64, ndim=1] bias, tuple stride):
	cdef unsigned int kernel_size0 = kernels.shape[1], kernel_size1 = kernels.shape[2], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = kernels.shape[0], \
	num_filters = kernels.shape[3], \
	batch_size = inputv.shape[0]

	cdef unsigned int out_size0 = (inputv.shape[1] - kernel_size0) / stride0 + 1, \
	out_size1 = (inputv.shape[2] - kernel_size1) / stride1 + 1

	cdef np.ndarray[FLOAT64, ndim=4] out = np.empty(shape=[batch_size, out_size0, out_size1, num_filters], dtype=np.float64)
	#np.ndarray[FLOAT64, ndim=4]
	cdef unsigned int b, i, j, m, kw, kh, n
	cdef unsigned int iin, jin
	cdef double acc	
	
	for b in prange(batch_size, nogil=True):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for n in range(num_filters):
					out[b, i, j, n] = 0.
					for kw in range(kernel_size0):
						for kh in range(kernel_size1):
							for m in range(num_dim):
								out[b, i, j, n] += inputv[b, iin + kw, jin + kh, m] * kernels[m, kw, kh, n]
					out[b, i, j, n] += bias[n]
	return out


# derivatives
@cython.wraparound(False)
@cython.boundscheck(False)
def nb_derivatives(np.ndarray[FLOAT64, ndim=4] doutput, np.ndarray[FLOAT64, ndim=4] inputv, np.ndarray[FLOAT64, ndim=4] kernels, tuple stride):
	cdef unsigned int out_size0 = doutput.shape[1], out_size1 = doutput.shape[2], \
	kernel_size0 = kernels.shape[1], kernel_size1 = kernels.shape[2], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = kernels.shape[0], \
	num_filters = kernels.shape[3], \
	batch_size = inputv.shape[0]

	cdef np.ndarray[FLOAT64, ndim=4] dx = np.zeros(shape=[batch_size, inputv.shape[1], inputv.shape[2], num_dim])
	cdef np.ndarray[FLOAT64, ndim=4] dw = np.zeros(shape=[num_dim, kernel_size0, kernel_size1, num_filters])
	cdef np.ndarray[FLOAT64, ndim=1] db = np.zeros(shape=[num_filters])

	cdef unsigned int b, i, j, m, kw, kh, n
	cdef unsigned int iin, jin
	cdef unsigned int iin_kw, jin_kh
	cdef double doutput_ptr

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for n in range(num_filters):
					doutput_ptr = doutput[b, i, j, n]
					#pragma omp critical
					db[n] += doutput_ptr

					for kw in range(kernel_size0):
						for kh in range(kernel_size1):
							iin_kw = (iin + kw)
							jin_kh = (jin + kh)

							for m in range(num_dim):
								#pragma omp critical
								dw[m, kw, kh, n] += inputv[b, iin_kw, jin_kh, m]*doutput_ptr
								dx[b, iin_kw, jin_kh, m] += kernels[m, kw, kh, n]*doutput_ptr
	return dx, dw, db