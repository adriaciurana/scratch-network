import hashlib
import importlib
import numpy as np

class Misc:
	@staticmethod
	def check_hash(module, classname, hash):
		return Misc.hash(module, classname) == hash
	def hash(module, classname):
		return hashlib.sha256((module + classname).encode('utf-8')).hexdigest()
	@staticmethod
	def import_class(module, classname):
		module = importlib.import_module(module)
		my_class = getattr(module, classname)
		return my_class

	def pack_hdf_name(name):
		return name.replace('/', '%2F')

	def unpack_hdf_name(name):
		return name.replace('%2F', '/')

	