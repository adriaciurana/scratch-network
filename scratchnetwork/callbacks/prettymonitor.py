import time, datetime, sys
from ..utils.prettyresults import PrettyResults
from .callback import Callback
from collections import OrderedDict 
class PrettyMonitor(Callback):
	TRAINING, VALIDATION, TRAINING_AND_VALIDATION = range(3)
	ESTIMATION_NONE, ESTIMATION_AVG, ESTIMATION_WINDOW = range(3)

	def __init__(self, split=None, iterations=None, estimated_time_finish=None, alpha_estimation_finish=0.9):
		if split is None:
			split = self.TRAINING
		
		if estimated_time_finish is None:
			estimated_time_finish = self.ESTIMATION_WINDOW
		
		self.iterations = iterations
		self.pretty = PrettyResults(fullscreen=True)
		self.split = split
		self.type_estimation_finish = estimated_time_finish

		# Inicializamos el alpha si hay ventana
		if self.type_estimation_finish == self.ESTIMATION_WINDOW:
			self.alpha_estimation_finish = alpha_estimation_finish

	def init(self, network):
		super(PrettyMonitor, self).init(network)
		self.iterations_monitoring = 0
		self.time_monitoring = 0

		# Inicializamos la estimacion de tiempo final
		if self.type_estimation_finish == self.ESTIMATION_AVG:
			self.time_monitoring_acc = 0
		
		elif self.type_estimation_finish == self.ESTIMATION_WINDOW:
			self.time_monitoring_acc = None

		# Inicializamos metricas y losses
		self.losses_monitoring_acc = [0 for _ in network.losses]
		self.metrics_monitoring_acc = [0 for _ in network.metrics]

	def __reset(self):
		self.iterations_monitoring = 0
		self.time_monitoring = 0

		for k in range(len(self.losses_monitoring_acc)):
			self.losses_monitoring_acc[k] = 0
		for k in range(len(self.metrics_monitoring_acc)):
			self.metrics_monitoring_acc[k] = 0

	def excecute_pre_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs):
		self.start_time = time.time()

	def excecute_post_batch(self, split, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs):
		elapsed_time = time.time() - self.start_time
		if (split == self.network.SPLIT.TRAINING and self.split == self.TRAINING) or \
			(split == self.network.SPLIT.VALIDATION and self.split == self.VALIDATION) or \
			self.split == self.TRAINING_AND_VALIDATION:

			for k in range(len(self.losses_monitoring_acc)):
				self.losses_monitoring_acc[k] += self.network.losses[k].temp_forward_result
			for k in range(len(self.metrics_monitoring_acc)):
				self.metrics_monitoring_acc[k] += self.network.metrics[k].temp_forward_result
			self.iterations_monitoring += 1
			self.time_monitoring += elapsed_time

			if self.type_estimation_finish == self.ESTIMATION_AVG:
				self.time_monitoring_acc += elapsed_time
			
			elif self.type_estimation_finish == self.ESTIMATION_WINDOW:
				if self.time_monitoring_acc == None:
					self.time_monitoring_acc = elapsed_time
				
				else:
					self.time_monitoring_acc = self.alpha_estimation_finish*self.time_monitoring_acc + (1 - self.alpha_estimation_finish)*elapsed_time

			if (self.iterations is not None and iteration > 0 and (iteration + 1) % self.iterations == 0) or (iteration  + 1) >= total_iterations:
				end_time = time.time()
				
				losses = dict([(self.network.losses[k].name, self.losses_monitoring_acc[k] / self.iterations_monitoring) for k in range(len(self.losses_monitoring_acc))])
				metrics = dict([(self.network.metrics[k].name, self.metrics_monitoring_acc[k] / self.iterations_monitoring) for k in range(len(self.metrics_monitoring_acc))])
				
				self.pretty.reset()
				if split == self.network.SPLIT.TRAINING:
					self.pretty.add_row('TRAINING')
				else:
					self.pretty.add_row('VALIDATION')
				
				self.pretty.add_row('Losses', '=')
				self.pretty.add_dictionary(losses, '-')
				
				self.pretty.add_row('Metrics', '=')
				self.pretty.add_dictionary(metrics, '-')
				
				self.pretty.add_row('Utils', '=')

				total_batchs_str = str(total_batchs)
				batch_index_str = str(batch_index + batch_size)
				batch_index_str = (len(total_batchs_str) - len(batch_index_str))*' ' + batch_index_str
				
				total_epochs_str = str(total_epochs)
				epoch_str = str(epoch + 1)
				epoch_str = (len(total_epochs_str) - len(epoch_str))*' ' + epoch_str

				total_iterations_str = str(total_iterations)
				iteration_str = str(iteration + 1)
				iteration_str = (len(total_iterations_str) - len(iteration_str))*' ' + iteration_str
				
				dict_info = OrderedDict()
				dict_info['Batch'] = batch_index_str + "/" + total_batchs_str
				dict_info['Epoch'] = epoch_str + "/" + total_epochs_str 
				dict_info['Iteration'] = iteration_str + "/" + total_iterations_str 
				dict_info['Elapsed Time'] =  '{:0.2f}'.format(self.time_monitoring) + " seconds / " + str(self.iterations) + " iterations"			
				self.pretty.add_dictionary(dict_info, '-')

				# Estimacion de tiempo para acabar
				if split == self.network.SPLIT.TRAINING and self.type_estimation_finish != self.ESTIMATION_NONE:
					if self.type_estimation_finish == self.ESTIMATION_AVG:
						estimated_time_finish = (total_iterations - (iteration + 1))*(self.time_monitoring_acc / iteration)
						estimated_time_finish_epochs = (((total_epochs - (epoch + 1))*total_iterations) - (iteration + 1))*(self.time_monitoring_acc / iteration)
					elif self.type_estimation_finish == self.ESTIMATION_WINDOW:
						estimated_time_finish = (total_iterations - (iteration + 1))*self.time_monitoring_acc
						estimated_time_finish_epochs = (((total_epochs - (epoch + 1))*total_iterations) - (iteration + 1))*self.time_monitoring_acc
					
					self.pretty.add_row('Estimations (doesn\'t count validation split)', '=')
					dict_estimation = OrderedDict()
					dict_estimation['Time to finish epoch'] = str(datetime.timedelta(seconds=estimated_time_finish))
					dict_estimation['Time to finish process'] = str(datetime.timedelta(seconds=estimated_time_finish_epochs))
					self.pretty.add_dictionary(dict_estimation)
					#self.pretty.add_row()
				self.pretty.print()

				self.__reset()