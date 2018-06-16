# max
def nb_forward_max(double[:, :, :, :] inputv, short[:, :, :, :, :] mask, double[:, :, :, :] out, 
					  tuple out_size, tuple pool_size, tuple stride):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], \
	pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]
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
							blockInput = inputv[b, iin + kw, jin + kh, m]
							if maxi == -1 or maxv < blockInput:
								maxv = blockInput
								maxi = kw
								maxj = kh
					mask[b, i, j, m, 0] = maxi
					mask[b, i, j, m, 1] = maxj
					out[b, i, j, m] = maxv

def nb_derivatives_max(short[:, :, :, :, :] mask, double[:, :, :, :] doutput,
				   double [:, :, :, :] dx,
				   tuple out_size, tuple stride):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = doutput.shape[3], \
	batch_size = doutput.shape[0]

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


# mean

def nb_forward_mean(double[:, :, :, :] inputv, double[:, :, :, :] out, 
					  tuple out_size, tuple pool_size, tuple stride):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], \
	pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = inputv.shape[3], \
	batch_size = inputv.shape[0]
	
	cdef int b, i, j, m, kw, kh, n
	cdef int idx, iin, jin
	cdef double blockInput
	cdef double mean
	cdef double den = pool_size0*pool_size1

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_dim):
					mean = 0.
					for kw in range(pool_size0):
						for kh in range(pool_size1):
							mean += inputv[b, iin + kw, jin + kh, m]
					out[b, i, j, m] = mean / den

def nb_derivatives_mean(double[:, :, :, :] doutput,
				   double [:, :, :, :] dx,
				   tuple out_size, tuple pool_size, tuple stride):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], \
	pool_size0 = pool_size[0], pool_size1 = pool_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = doutput.shape[3], \
	batch_size = doutput.shape[0]

	cdef int b, i, j, m, n
	cdef int idx, iin, jin
	cdef double blockInput
	cdef int iin_kw, jin_kh
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

