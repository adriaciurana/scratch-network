from .optimizer import Optimizer
import numpy as np
class SGD(Optimizer):
	def __init__(self, lr=1e-3, mu=0.9, clip=1):
		self.lr = lr
		self.mu = mu
		self.clip = clip
		self.iweights = {}
	
	def step(self, label, weight_name, dweight):
		weight_name = str(label) + '_' + weight_name
		if self.clip is None:
			coef = 1
		else:
			dweight_flatten = dweight.flatten()
			dweight_norm = np.linalg.norm(dweight_flatten, 2)
			coef = self.clip/max(dweight_norm, self.clip)

		try:
			iweight = self.iweights[weight_name]
		except:
			iweight = 0 
		self.iweights[weight_name] = -self.lr*dweight*coef + self.mu*iweight
		return self.iweights[weight_name]

	def save(self, h5_container):
		layer_json = super(SGD, self).save(h5_container)
		layer_json['attributes']['lr'] = self.lr
		layer_json['attributes']['mu'] = self.mu
		layer_json['attributes']['clip'] = self.clip
		h5_iweights = h5_container.create_group('iweights')
		for k, v in self.iweights.items():
			h5_iweights.create_dataset(k, data=v, dtype=v.dtype)
		return layer_json
		
	def load(self, data, h5_container):
		super(SGD, self).load(data, h5_container)
		self.lr = data['attributes']['lr']
		self.mu = data['attributes']['mu']
		self.clip = data['attributes']['clip']
		h5_iweights = h5_container['iweights']
		self.iweights = {}
		for k in h5_iweights.keys():
			self.iweights[k] = h5_iweights[k].value
		
