import numpy as np
from scipy.stats import truncnorm
class Initializer(object):
	FAN_IN, FAN_OUT, FAN_AVG = range(3)
	NORMAL, UNIFORM = range(2)
	
	def __init__(self, name, *args):
		self.name = name
		self.args = args

	def get(self, shape):
		def compute_truncated_normal(shape, scale):
			return truncnorm.rvs(a= (-2. / stddev), b= (2. / stddev), scale=stddev, loc=0., size=shape)
		
		def compute_fans(shape):
			# Keras function
			if len(shape) == 2:
				fan_in = shape[0]
				fan_out = shape[1]
			elif len(shape) in {3, 4, 5}:
				receptive_field_size = np.prod(shape[1:-1])
				fan_in = shape[0]*receptive_field_size
				fan_out = shape[-1]*receptive_field_size
			else:
				fan_in = np.sqrt(np.prod(shape))
				fan_out = np.sqrt(np.prod(shape))
			return fan_in, fan_out

		def compute_mode(shape, mode, scale):
			fan_in, fan_out = compute_fans(shape)
			if mode == Initializer.FAN_IN:
				scale /= max(1., fan_in)
			if mode == Initializer.FAN_OUT:
				scale /= max(1., fan_in)
			if mode == Initializer.FAN_AVG:
				scale /= max(1., float(fan_in + fan_out) / 2)
			return scale

		def compute_distribution(shape, distribution=Initializer.UNIFORM, mode=Initializer.FAN_IN, scale=1.0):
			scale = compute_mode(shape, mode, scale)
			if distribution == Initializer.NORMAL:
				stddev = np.sqrt(scale)
				return np.float64(compute_truncated_normal(shape, scale))
			elif distribution == Initializer.UNIFORM:
				stddev = np.sqrt(3.*scale)
				return np.float64(-stddev + np.random.rand(*shape)*(2*stddev))


		
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

		elif self.name == 'he':
			if len(self.args) == 0 or (len(self.args) == 1 and self.args[0] == 'uniform'):
				return compute_distribution(shape, distribution=Initializer.UNIFORM, mode=Initializer.FAN_IN, scale=2.0)
			elif len(self.args) == 1 and self.args[0] == 'normal':
				return compute_distribution(shape, distribution=Initializer.NORMAL, mode=Initializer.FAN_IN, scale=2.0)

		elif self.name == 'lecun':
			if len(self.args) == 0 or (len(self.args) == 1 and self.args[0] == 'uniform'):
				return compute_distribution(shape, distribution=Initializer.UNIFORM, mode=Initializer.FAN_IN, scale=1.0)
			elif len(self.args) == 1 and self.args[0] == 'normal':
				return compute_distribution(shape, distribution=Initializer.NORMAL, mode=Initializer.FAN_IN, scale=1.0)

		elif self.name == 'glorot':
			if len(self.args) == 0 or (len(self.args) == 1 and self.args[0] == 'uniform'):
				return compute_distribution(shape, distribution=Initializer.UNIFORM, mode=Initializer.FAN_AVG, scale=1.0)
			elif len(self.args) == 1 and self.args[0] == 'normal':
				return compute_distribution(shape, distribution=Initializer.NORMAL, mode=Initializer.FAN_AVG, scale=1.0)