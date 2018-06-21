from ..layers.layer import Layer
import numpy as np
class Loss(Layer):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		params['same_input_shape'] = True
		params['compute_forward_in_prediction'] = False
		super(Loss, self).__init__(node, params=params)
		self.weight = 1

	def computeSize(self):
		return tuple([1])
		
	def forward(self, inputs):
		super(Loss, self).forward(inputs)

	def derivatives(self, doutput=None):
		pass

	def save(self, h5_container):
		layer_json = super(Loss, self).save(h5_container)
		layer_json['attributes']['weight'] = self.weight
		return layer_json
		
	def load(self, data, h5_container):
		super(Loss, self).load(data, h5_container)
		self.weight = data['attributes']['weight']
		