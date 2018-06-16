# forward
def nb_forward(double[:, :, :, :] input, double[:, :, :, :] kernels, double[:] bias, double[:, :, :, :] out, 
					  tuple out_size, tuple kernel_size, tuple stride):
	cdef int out_size0 = out_size[0], \
	out_size1 = out_size[1], \
	kernel_size0 = kernel_size[0], kernel_size1 = kernel_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = kernels.shape[0], num_filters = kernels.shape[3], \
	batch_size = input.shape[0]

	cdef int b, i, j, m, kw, kh, n
	cdef int iin, jin
	cdef double acc

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for n in range(num_filters):
					acc = 0.
					for m in range(num_dim):	
						for kw in range(kernel_size0):
							for kh in range(kernel_size1):
								acc += input[b, iin + kw, jin + kh, m] * kernels[m, kw, kh, n]
					out[b, i, j, n] = acc + bias[n]


# derivatives
def nb_derivatives(double[:, :, :, :] input, double[:, :, :, :] doutput, double[:, :, :, :] kernels,
				   double[:, :, :, :] dw, double [:] db, double [:, :, :, :] dx,
				   tuple out_size, tuple kernel_size, tuple stride):
	cdef int out_size0 = out_size[0], \
	out_size1 = out_size[1], \
	kernel_size0 = kernel_size[0], kernel_size1 = kernel_size[1], \
	stride0 = stride[0], stride1 = stride[1], \
	num_dim = kernels.shape[0], num_filters = kernels.shape[3], \
	batch_size = input.shape[0]

	cdef int b, i, j, m, kw, kh, n
	cdef int iin, jin
	cdef int iin_kw, jin_kh
	cdef double doutput_ptr

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for n in range(num_filters):
					doutput_ptr = doutput[b, i, j, n]
					db[n] += doutput_ptr

					for kw in range(kernel_size0):
						for kh in range(kernel_size1):
							iin_kw = (iin + kw)
							jin_kh = (jin + kh)

							for m in range(num_dim):
								dw[m, kw, kh, n] += input[b, iin_kw, jin_kh, m]*doutput_ptr
								dx[b, iin_kw, jin_kh, m] += kernels[m, kw, kh, n]*doutput_ptr