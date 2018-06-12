from ..losses.loss import Loss
import numpy as np
class Metric(Loss):
	def __init__(self, node, params={}):
		params['compute_backward'] = False
		
		# No produce dependencias ya que no participa en el forward (porque es un nodo final) ni participa en el backward porque es una metrica.
		super(Metric, self).__init__(node, params)

	def computeSize(self):
		return tuple([1])

	def forward(self, inputs):
		super(Loss, self).forward(inputs)
		pred, true = inputs
		out = np.reshape(pred, [pred.shape[0], -1]) - np.reshape(true, [true.shape[0], -1])
		return np.sqrt(np.mean(out**2, axis=0))
