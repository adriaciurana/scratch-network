from .layer import Layer
import numpy as np
class Concat(Layer):
	def __init__(self, node, axis=0, params=None):
		if params is None:
			params = {}
		super(Concat, self).__init__(node, params=params)
		self.axis = axis
	
	def computeSize(self):
		super(Concat, self).computeSize()
		out = [i for i in self.in_size[0]]
		out[self.axis] = 0
		
		self.accum_idx = []
		
		for in_size in self.in_size:
			aux = in_size[self.axis]
			out[self.axis] += aux
			self.accum_idx.append(aux)

		self.accum_idx = self.accum_idx[:-1]
		return tuple(out)

	def forward(self, inputs):
		super(Concat, self).forward(inputs)
		return np.concatenate(inputs, axis=self.axis + 1)

	def derivatives(self, doutput):
		a = np.split(doutput, self.accum_idx, axis=self.axis + 1)
		return np.split(doutput, self.accum_idx, axis=self.axis + 1), None