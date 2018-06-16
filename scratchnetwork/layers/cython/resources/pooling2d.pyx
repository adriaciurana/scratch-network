# forward
def nb_forward_max(double[:, :, :, :] input, short[:, :, :, :, :] mask, double[:, :, :, :] out, 
					  tuple out_size, tuple pool_size, tuple stride, 
					  int batch_size, int num_dim):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], pool_size0 = pool_size[0], pool_size1 = pool_size[1], stride0 = stride[0], stride1 = stride[1]
	cdef double INFINITY = float('-inf')

	cdef int b, i, j, m, kw, kh, n
	cdef int idx, iin, jin
	cdef double blockInput
	cdef double maxv
	cdef int maxi

	for b in range(batch_size):
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
							blockInput = input[b, iin + kw, jin + kh, m]
							if maxi == -1 or maxv < blockInput:
								maxv = blockInput
								maxi = kw
								maxj = kh
					mask[b, i, j, m, 0] = maxi
					mask[b, i, j, m, 1] = maxj
					out[b, i, j, m] = maxv


# derivatives
def nb_derivatives_max(short[:, :, :, :, :] mask, double[:, :, :, :] doutput,
				   double [:, :, :, :] dx,
				   tuple out_size, tuple stride, 
				   int batch_size, int num_dim):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], stride0 = stride[0], stride1 = stride[1]

	cdef int b, i, j, m, n
	cdef int idx, iin, jin
	cdef double blockInput
	cdef int iin_kw, jin_kh

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					iin_kw = iin + mask[b, i, j, m, 0]
					jin_kh = jin + mask[b, i, j, m, 1]
					dx[b, iin_kw, jin_kh, m] += doutput[b, i, j, m]