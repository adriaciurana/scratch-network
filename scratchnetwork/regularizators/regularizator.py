class Regularizator(object):
	def __init__(self, lambda_value=0.0005):
		self.lambda_value = lambda_value

	def function(self):
		raise NotImplemented