from .layer import Layer
import numpy as np
class DropOut(Layer):
	def __init__(self, node, prob, params={}):
		params['number_of_inputs'] = 1
		super(DropOut, self).__init__(node, params=params)
		self.prob = prob
	
	def computeSize(self):
		super(DropOut, self).computeSize()
		return tuple(self.in_size[0])

	def forward(self, inputs):
		super(DropOut, self).forward(inputs)
		input = inputs[0]

		if not self.predict_flag:
			self.values.mask = np.random.binomial(1, self.prob, size=input.shape) / self.prob
		else:
			self.values.mask = 1./self.prob
		return input * self.values.mask

	def derivatives(self, doutput):
		return doutput*self.values.mask