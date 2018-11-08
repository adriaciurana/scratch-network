import time
from ..utils.prettyresults import PrettyResults
from .callback import Callback
from collections import OrderedDict 
class PrettyMonitor(Callback):
	TRAINING, VALIDATION, TRAINING_AND_VALIDATION = range(3) 

	def __init__(self, split=None, iterations=None):
		if split is None:
			split = PrettyMonitor.TRAINING
		self.iterations = iterations
		self.pretty = PrettyResults(fullscreen=True)
		self.split = split

	def init(self, network):
		super(PrettyMonitor, self).init(network)
		self.iterations_monitoring = 0
		self.time_monitoring = 0
		self.losses_monitoring = [0 for _ in network.losses]
		self.metrics_monitoring = [0 for _ in network.metrics]

	def __reset(self):
		self.iterations_monitoring = 0
		self.time_monitoring = 0
		for k in range(len(self.losses_monitoring)):
			self.losses_monitoring[k] = 0
		for k in range(len(self.metrics_monitoring)):
			self.metrics_monitoring[k] = 0

	def excecute_pre_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs):
		pass

	def excecute_post_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs, elapsed_time):
		if (split == self.network.SPLIT.TRAINING and self.split == PrettyMonitor.TRAINING) or \
			(split == self.network.SPLIT.VALIDATION and self.split == PrettyMonitor.VALIDATION) or \
			self.split == PrettyMonitor.TRAINING_AND_VALIDATION:

			for k in range(len(self.losses_monitoring)):
				self.losses_monitoring[k] += self.network.losses[k].temp_forward_result
			for k in range(len(self.metrics_monitoring)):
				self.metrics_monitoring[k] += self.network.metrics[k].temp_forward_result
			self.iterations_monitoring += 1
			self.time_monitoring += elapsed_time

			if (self.iterations is not None and iteration > 0 and (iteration + 1) % self.iterations == 0) or (iteration  + 1) >= total_iterations:
				end_time = time.time()
				
				losses = dict([(self.network.losses[k].name, self.losses_monitoring[k] / self.iterations_monitoring) for k in range(len(self.losses_monitoring))])
				metrics = dict([(self.network.metrics[k].name, self.metrics_monitoring[k] / self.iterations_monitoring) for k in range(len(self.metrics_monitoring))])
				
				self.pretty.reset()
				if split == self.network.SPLIT.TRAINING:
					self.pretty.add_row('TRAINING')
				else:
					self.pretty.add_row('VALIDATION')
				
				self.pretty.add_row()
				self.pretty.add_row('Losses', '=')
				self.pretty.add_dictionary(losses, '-')
				
				self.pretty.add_row()
				self.pretty.add_row('Metrics', '=')
				self.pretty.add_dictionary(metrics, '-')
				
				self.pretty.add_row()
				self.pretty.add_row('Utils', '=')

				total_batchs_str = str(total_batchs)
				batch_index_str = str(batch_index)
				batch_index_str = (len(total_batchs_str) - len(batch_index_str))*' ' + batch_index_str
				
				total_epochs_str = str(total_epochs)
				epoch_str = str(epoch)
				epoch_str = (len(total_epochs_str) - len(epoch_str))*' ' + epoch_str

				total_iterations_str = str(total_iterations)
				iteration_str = str(iteration)
				iteration_str = (len(total_iterations_str) - len(iteration_str))*' ' + iteration_str
				
				dict_info = OrderedDict()
				dict_info['Batch'] = batch_index_str + "/" + total_batchs_str
				dict_info['Epoch'] = epoch_str + "/" + total_epochs_str 
				dict_info['Iteration'] = iteration_str + "/" + total_iterations_str 
				dict_info['Elapsed Time'] =  '{:0.2f}'.format(self.time_monitoring)
				
				self.pretty.add_dictionary(dict_info, '-')
				self.pretty.print()

				self.__reset()