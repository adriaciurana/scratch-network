from .layer import Layer
import numpy as np
class PReLU(Layer):
	def __init__(self, node, alpha, params=None):
		if params is None:
			params = {}
		self.alpha = alpha
			
		params['number_of_inputs'] = 1
		super(PReLU, self).__init__(node, params=params)
	
	def computeSize(self):
		super(PReLU, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(PReLU, self).forward(inputs)
		input = inputs[0]

		self.values.input = np.array(input, copy=True)
		xindex = self.values.input < 0
		self.values.input[xindex] = self.alpha*self.values.input[xindex]
		return self.values.input

	def derivatives(self, doutput):
		dx = np.array(doutput, copy=True)
		dx[self.values.input <= 0] = self.alpha
		return dx
  



