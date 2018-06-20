from .layer import Layer
import numpy as np
class ReLU(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		params['number_of_inputs'] = 1
		super(ReLU, self).__init__(node, params=params)
	
	def computeSize(self):
		super(ReLU, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(ReLU, self).forward(inputs)
		input = inputs[0]

		self.values.input = inputs[0]
		return np.maximum(0, self.values.input)

	def derivatives(self, doutput):
		dx = np.array(doutput, copy=True)
		dx[self.values.input <= 0] = 0
		return dx
  



