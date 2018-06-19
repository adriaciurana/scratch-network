import numpy as np
from ..backend.misc import Misc
from ..optimizers import *
class Optimizer(object):
	def __init__(self):
		raise NotImplemented
	
	def step(self, label, weight_name, dweight):
		raise NotImplemented

	def save(self, h5_container):
		optimizer_json = {'type': self.__class__.__name__, 'module': self.__class__.__module__, 'attributes':{}}
		optimizer_json['hash'] = Misc.hash(optimizer_json['module'], optimizer_json['type'])
		return optimizer_json

	def load(self, data, h5_container):
		pass

	@staticmethod
	def load_static(data, h5_container):
		if not Misc.check_hash(data['module'], data['type'], data['hash']):
			raise IndexError # Error
		my_class = Misc.import_class(data['module'], data['type'])
		obj = my_class.__new__(my_class)
		
		obj.load(data, h5_container)
		return obj


		

