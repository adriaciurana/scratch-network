from .optimizer import Optimizer
import numpy as np
from ..backend.misc import Misc
class SGD(Optimizer):
	# Elementos que se deben inicializar siempre
	def init(self):
		super(SGD, self).init(['iweights'])
		self.iweights = {}

	def __init__(self, lr=1e-2, mu=0.9, clip_norm=1, decay=0., nesterov=False):
		super(SGD, self).__init__()
		self.lr = lr
		self.decay = decay
		self.mu = mu
		self.clip_norm = clip_norm
		self.nesterov = nesterov
	
	def step(self, label, weight_name, dweight):
		weight_name = str(label) + '_' + weight_name
		if self.clip_norm is not None:
			dweight_flatten = dweight.flatten()
			dweight_norm = np.linalg.norm(dweight_flatten, 2)
			dweight = self.clip_norm/max(dweight_norm, self.clip_norm)*dweight
		
		if self.iterations == 0 or self.mu == 0:
			iweight = 0
		else:
			iweight = self.iweights[weight_name]

		lr = self.lr
		if self.decay > 0:
			lr *= (1. / (1. + self.decay * self.iterations))

		iweight = -lr*dweight + self.mu*iweight
		if self.nesterov:
			iweight =  -lr *dweight + self.mu*iweight
		
		self.iweights[weight_name] = iweight
		return iweight

	def save(self, h5_container):
		layer_json = super(SGD, self).save(h5_container)
		layer_json['attributes']['lr'] = self.lr
		layer_json['attributes']['decay'] = self.decay
		layer_json['attributes']['mu'] = self.mu
		layer_json['attributes']['clip_norm'] = self.clip_norm
		layer_json['attributes']['nesterov'] = self.nesterov
		h5_iweights = h5_container.create_group('iweights')
		for k, v in self.iweights.items():
			h5_iweights.create_dataset(Misc.pack_hdf_name(k), data=v, dtype=v.dtype)
		return layer_json
		
	def load(self, data, h5_container):
		super(SGD, self).load(data, h5_container)
		self.lr = data['attributes']['lr']
		self.decay = data['attributes']['decay']
		self.mu = data['attributes']['mu']
		self.clip_norm = data['attributes']['clip_norm']
		self.nesterov = data['attributes']['nesterov']
		h5_iweights = h5_container['iweights']
		self.iweights = {}
		for k in h5_iweights.keys():
			self.iweights[Misc.unpack_hdf_name(k)] = h5_iweights[k].value
		
