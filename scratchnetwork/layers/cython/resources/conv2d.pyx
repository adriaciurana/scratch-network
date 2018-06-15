# forward
def nb_forward(double[:, :, :, :] input, double[:, :] kernels, double[:, :] bias, double[:, :, :, :] out, 
					  tuple out_size, tuple kernel_size, tuple stride, 
					  int batch_size, int num_dim, int num_filters):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], kernel_size0 = kernel_size[0], kernel_size1 = kernel_size[1], stride0 = stride[0], stride1 = stride[1]

	cdef int b, i, j, m, kw, kh, n
	cdef int idx, iin, jin
	cdef double acc, blockInput

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_filters):
					acc = 0.
					for kw in range(kernel_size0):
						for kh in range(kernel_size1):
							blockInput = input[b, iin + kw, jin + kh, m]
							for n in range(num_dim):
								idx = kw*kernel_size1*num_dim + kh*num_dim + n #idx = np.ravel_multi_index([kw, kh, n], [num_dim, kernel_size[1], kernel_size[0]])
								acc += blockInput * kernels[idx, m]
					out[b, i, j, m] = acc + bias[0, m]


# derivatives
def nb_derivatives(double[:, :, :, :] input, double[:, :, :, :] doutput, double[:, :] kernels, double[:, :] bias,
				   double[:, :] dw, double [:, :] db, double [:, :, :, :] dx,
				   tuple out_size, tuple kernel_size, tuple stride, 
				   int batch_size, int num_dim, int num_filters):
	cdef int out_size0 = out_size[0], out_size1 = out_size[1], kernel_size0 = kernel_size[0], kernel_size1 = kernel_size[1], stride0 = stride[0], stride1 = stride[1]

	cdef int b, i, j, m, kw, kh, n
	cdef int idx, iin, jin
	cdef double acc, blockInput
	cdef int iin_kw, jin_kh
	cdef double doutput_ptr

	for b in range(batch_size):
		for i in range(out_size0):
			for j in range(out_size1):
				iin = i*stride0
				jin = j*stride1

				for m in range(num_filters):
					acc = 0.
					for kw in range(kernel_size0):
						for kh in range(kernel_size1):
							iin_kw = (iin + kw)
							jin_kh = (jin + kh)
							blockInput = input[b, iin_kw, jin_kh, m]

							for n in range(num_dim):
								idx = kw*kernel_size1*num_dim + kh*num_dim + n #idx = np.ravel_multi_index([kw, kh, n], [num_dim, kernel_size[1], kernel_size[0]])
								doutput_ptr = doutput[b, i, j, n]

								dw[idx, m] += blockInput*doutput_ptr
								db[0, m] += doutput_ptr
								dx[b, iin_kw, jin_kh, m] += kernels[idx, m]*doutput_ptr