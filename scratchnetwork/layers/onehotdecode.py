from .layer import Layer
import numpy as np
class OneHotDecode(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}

		params['compute_backward'] = False
		params['number_of_inputs'] = 1
		super(OneHotDecode, self).__init__(node, params=params)
	
	def computeSize(self):
		super(OneHotDecode, self).computeSize()
		return tuple([1])

	def forward(self, inputs):
		super(OneHotDecode, self).forward(inputs)
		return np.argmax(inputs[0], axis=-1).reshape(-1, 1)