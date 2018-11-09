from .optimizer import Optimizer
import numpy as np
from ..backend.misc import Misc
class AdaGrad(Optimizer):
	# Elementos que se deben inicializar siempre
	def init(self):
		super(AdaGrad, self).init(['g2'])
		self.g2 = {}

	def __init__(self, lr=1e-2, clip_norm=1, decay=0.):
		super(AdaGrad, self).__init__()
		self.init()
		self.lr = lr
		self.decay = decay
		self.clip_norm = clip_norm
	
	def step(self, label, weight_name, dweight):
		weight_name = str(label) + '_' + weight_name
		if self.clip_norm is not None:
			dweight_flatten = dweight.flatten()
			dweight_norm = np.linalg.norm(dweight_flatten, 2)
			dweight = self.clip_norm/max(dweight_norm, self.clip_norm)*dweight
		
		if self.iterations == 0:
			g2 = 0
		else:
			g2 = self.g2[weight_name]

		lr = self.lr
		if self.decay > 0:
			lr *= (1. / (1. + self.decay * self.iterations))

		g2 += dweight**2
		self.g2[weight_name] = g2

		iweight = -lr*dweight / np.sqrt(g2 + 10e-8)
		return iweight

	def save(self, h5_container):
		layer_json = super(AdaGrad, self).save(h5_container)
		layer_json['attributes']['lr'] = self.lr
		layer_json['attributes']['decay'] = self.decay
		layer_json['attributes']['clip_norm'] = self.clip_norm
		h5_g2 = h5_container.create_group('g2')
		for k, v in self.g2.items():
			h5_g2.create_dataset(Misc.pack_hdf_name(k), data=v, dtype=v.dtype)
		return layer_json
		
	def load(self, data, h5_container):
		super(AdaGrad, self).load(data, h5_container)
		self.lr = data['attributes']['lr']
		self.decay = data['attributes']['decay']
		self.clip_norm = data['attributes']['clip_norm']
		h5_g2 = h5_container['g2']
		self.g2 = {}
		for k in h5_g2.keys():
			self.g2[Misc.unpack_hdf_name(k)] = h5_g2[k].value
		
