class Pipeline(object):
	def __init__(self):
		pass

	def addPrev(self):
		raise NotImplementedError

	def addNext(self):
		raise NotImplementedError

	@property
	def end(self):
		raise NotImplementedError

	@property
	def start(self):
		raise NotImplementedError