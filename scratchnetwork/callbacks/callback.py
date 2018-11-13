class Callback(object):
	def __init__(self):
		raise NotImplementedError
	
	def init(self, network):
		self.network = network

	def excecute_pre_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs):
		pass
	
	def excecute_post_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs):
		pass