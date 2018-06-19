from ..backend.misc import Misc
class Regularizator(object):
	def __init__(self, lambda_value=0.0005):
		self.lambda_value = lambda_value

	def function(self):
		raise NotImplemented

	def save(self, h5_container):
		optimizer_json = {'type': self.__class__.__name__, 'module': self.__class__.__module__, 'attributes':{}}
		optimizer_json['hash'] = Misc.hash(optimizer_json['module'], optimizer_json['type'])
		optimizer_json['attributes']['lambda'] = self.lambda_value
		return optimizer_json

	def load(self, data, h5_container):
		pass

	@staticmethod
	def load_static(data, h5_container):
		if not Misc.check_hash(data['module'], data['type'], data['hash']):
			raise IndexError # Error
		my_class = Misc.import_class(data['module'], data['type'])
		obj = my_class.__new__(my_class)
		obj.lambda_value = data['attributes']['lambda']
		obj.load(data, h5_container)
		return obj