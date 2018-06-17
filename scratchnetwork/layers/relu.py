from .layer import Layer
import numpy as np
from .cython import relu3d
from .cython import relu2d
from .cython import relu1d
from .cython import relu
class ReLU(Layer):
	def __init__(self, node, params={}):
		params['number_of_inputs'] = 1
		super(ReLU, self).__init__(node, params=params)
	
	def computeSize(self):
		super(ReLU, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(ReLU, self).forward(inputs)
		input = inputs[0]

		if len(self.in_size[0]) == 3:
			self.values.input = relu3d.nb_forward(input)
			return self.values.input

		elif len(self.in_size[0]) == 2:
			self.values.input = relu2d.nb_forward(input)
			return self.values.input

		elif len(self.in_size[0]) == 1:
			self.values.input = relu1d.nb_forward(input)
			return self.values.input
		else:
			self.values.input = relu.nb_forward(input.flatten())
			return self.values.input.reshape(input.shape)


		"""input = abs(input) * (input > 0)
		self.values.input = input
		return input"""

	def derivatives(self, doutput):
		"""partial = self.values.input > 0
		return doutput*partial"""
		if len(self.out_size) == 3:
			return relu3d.nb_derivatives(doutput, self.values.input)

		elif len(self.out_size) == 2:
			return relu2d.nb_derivatives(doutput, self.values.input)

		elif len(self.out_size) == 1:
			return relu1d.nb_derivatives(doutput, self.values.input)

		else:
			dx = relu.nb_derivatives(doutput.flatten(), self.values.input)
			return dx.reshape(doutput.shape)



