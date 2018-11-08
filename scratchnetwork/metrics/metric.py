from ..layers.avoidfreeze import AvoidFreeze
import numpy as np
class Metric(AvoidFreeze):
	def __init__(self, node, params=None):
		if params is None:
			params = {}
			
		params['compute_backward'] = False	
		params['same_input_shape'] = True
		params['compute_forward_in_prediction'] = False
		# No produce dependencias ya que no participa en el forward (porque es un nodo final) ni participa en el backward porque es una metrica.
		super(Metric, self).__init__(node, params=params)

	def computeSize(self):
		return tuple([1])

	def forward(self, inputs):
		super(Metric, self).forward(inputs)

	def derivatives(self, doutput=None):
		pass