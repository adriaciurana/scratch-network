import numpy as np
from scipy.stats import truncnorm
class Initializer(object):
	
	def __init__(self, name, *args):
		self.name = name
		self.args = args

	def get(self, shape):
		def compute_truncated_normal(scale, shape):
			scale = 1./max(1, scale)
			stddev = np.sqrt(scale)
			return truncnorm.rvs(a= (-2. / stddev), b= (2. / stddev), scale=stddev, loc=0., size=shape)
		
		if self.name == 'ones':
			return np.ones(shape=shape)
		
		elif self.name == 'zeros':
			return np.zeros(shape=shape)
		
		elif self.name == 'constant':
			return np.ones(shape=shape)*self.args[0]
		
		elif self.name == 'rand':
			# 0-1 uniform
			if len(self.args) == 0:
				return np.rand(*shape)
			
			# 0-N
			elif len(self.args) == 1:
				return np.rand(*shape)*self.args[0]

			# a-b
			elif len(self.args) == 2:
				return self.args[0] + np.random.rand(*shape)*(self.args[1] - self.args[0])

		elif self.name == 'normal':
			if len(self.args) == 0:
				return np.random.randn(*shape)

			elif len(self.args) == 2:
				return np.random.randn(*shape)*self.args[1] + self.args[0]

		elif self.name == 'xavier':
			# receptive_field = w * h * ... (of filter, weights, etc)
			# fan_in = receptive_field * input_features
			fan_in = np.prod(shape[:-1])
			fan_out = np.prod(shape[1:])
			scale = float(fan_in + fan_out) / 2
			return compute_truncated_normal(scale, shape)

		elif self.name == 'lecun':
			# receptive_field = w * h * ... (of filter, weights, etc)
			# fan_in = receptive_field * input_features
			fan_in = np.prod(shape[:-1])
			scale = fan_in
			return compute_truncated_normal(scale, shape)	