import numpy as np
from .layers.layer import Layer
from .backend.exceptions import Exceptions

import time
class Node(object):
	INPUT, OUTPUT, MIDDLE, NOT_CONNECTED = range(4)
	def __init__(self, network, name, layer, layer_args, layer_kargs):
		self.temp_forward_dependences = 0
		self.temp_backward_dependences = 0
		
		self.network = network
		self.name = name

		# Parametros pre-creacion de la capa
		self.compute_forward_in_prediction = True

		self.compute_backward = True
		# indicamos si produce o no dependencia tanto en el forward como el backward, las metricas no bloquean las dependencias ya que no participan en el juego solo evaluan.
		
		if issubclass(layer, Layer):
			layer_args = tuple([self] + list(layer_args))
			self.layer = layer(*layer_args, **layer_kargs)
		else:
			raise Exceptions.NotFoundLayer("La capa que has introducido no existe.")

		# relaciones que tiene el nodo
		self.prevs = []
		self.nexts = []

	"""
		RELATIONS
	"""
	def addNext(self, node):
		self.nexts.append(node)
		node.prevs.append(self)


	def addPrev(self, node):
		self.prevs.append(node)
		node.nexts.append(self)

	# Las variables que empiezan por temp, son temporales y son las que se usan en el flujo del programa.
	"""
		FORWARD:
	"""
	def clearForwardDependences(self):
		self.temp_forward_dependences = 0

	def incrementForwardDependences(self):
		self.temp_forward_dependences += 1

	def checkForwardDependences(self):
		return self.temp_forward_dependences == len(self.prevs)
	
	def clearForwardResults(self):
		self.temp_forward_result = None

	def computeForward(self):
		inputs = [n.temp_forward_result for n in self.prevs]
		if self.network.firstForward:
			self.layer.firstForward(inputs)
		return self.layer.forward(inputs)
		
	def forward(self):
		# Si esta en modo no prediccion el forward se para en la Loss
		if self.network.predict_flag and not self.compute_forward_in_prediction:
			return

		result = self.computeForward()
		self.temp_forward_result = result
		for n in self.nexts:
			n.incrementForwardDependences()
			# Solo se podra ejecutar si todas las dependencias han terminado de calcularse.
			if n.checkForwardDependences():
				n.forward()
				n.clearForwardDependences()
		return result

	"""
		BACKWARD:
	"""
	def clearBackwardDependences(self):
		self.temp_backward_dependences = 0

	def incrementBackwardDependences(self):
		self.temp_backward_dependences += 1

	def checkBackwardDependences(self):
		return self.temp_backward_dependences == self.number_backward_sum_nexts_nodes #sum([n.compute_backward for n in self.nexts])
	
	def clearBackwardResults(self):
		self.temp_backward_result = None

	def computeBackwardAndCorrect(self, has_any_backward_to_compute):
		# Obtenemos las derivadas de las proximas salidas
		doutput = self.temp_backward_result
		if has_any_backward_to_compute:
			unpack_derivatives = self.layer.derivatives(doutput)
			if isinstance(unpack_derivatives, (list, tuple)):
				backward, dweights = unpack_derivatives
			else:
				backward = unpack_derivatives
				dweights = None

		else:
			backward = None
			dweights = None

		# Transferimos las contribuciones dentro de la capa para corregir los pesos internos
		# Es importante hacerlo despues del backward, porque este paso variara el valor de los pesos internos
		if self.layer.is_trainable and dweights is not None:
			self.layer.correctWeights(dweights)
		return backward
	
	def backpropagation(self, is_loss = False):
		# Si es una loss, debemos indicarlo en la capa posterior, podriamos hacerlo usando isinstance pero es mas rapido usar un booleano.
		# La diferencia entre un capa normal y una loss es que la loss debe añadirse al final de todo el proceso de la red
		# Eso es debido a que tiene una naturala distinta al contener los datos en un funcional.
		has_any_backward_to_compute = self.number_backward_any_prevs_nodes # any([n.compute_backward for n in self.prevs])

		#global t
		#t = time.time()
		backward = self.computeBackwardAndCorrect(has_any_backward_to_compute)
		#print(self.name, time.time() - t)

		# si no tiene backward o no tiene nodos a los que enviar nada, el proceso se termina
		if not has_any_backward_to_compute or backward is None:
			return
		
		# si es una loss multiplicamos el peso asociado
		if is_loss:
			backward *= self.network.weights_losses[self.layer]

		# propagamos hacia atras
		for i, n in enumerate(self.prevs):
			if n.compute_backward:
				# si se devuelve una lista, querrá decir que la derivada es distinta por cada entrada
				if isinstance(backward, list):
					b = backward[i]
				else:
					# la derivada es comun para todas las entradas
					b = backward

				# Si no se ha calculado aun el backward se inicializa con la primera contribucion
				# Esto sucede con nodos que su salida se conecta a distintos nodos, la derivada al final es la suma de todas las contribuciones.
				# dy/dx = dy/dx(salida1) + dy/dx(salida2) + ...
				if n.temp_backward_dependences == 0:
					n.temp_backward_result = b
				else:
					n.temp_backward_result += b

				# Solo se podra ejecutar si todas las dependencias han terminado de calcularse.
				n.incrementBackwardDependences()
				if n.checkBackwardDependences():
					n.backpropagation()
					# una vez realizado el backward se vuelven a reinicializar las estructuras
					# en el backward debemos volver a poner a 0 tambien 
					n.clearBackwardDependences()

	def computeNumberOfBackwardNodes(self):
		# para no tener que calcularlo en cada iteracion, lo calculamos en la compilacion
		self.number_backward_sum_nexts_nodes = sum([n.compute_backward for n in self.nexts])
		self.number_backward_any_prevs_nodes = any([n.compute_backward for n in self.prevs])

	""" 
		COMPUTE SIZE:
			Calcula el tamaño de la capa
	"""
	def clearSizesResults(self):
		self.layer.in_size = []
		self.layer.in_size_flatten = []
		self.layer.out_size = None
		self.layer.out_size_flatten = None

	def checkComputeSizeDependences(self):
		check = True
		for n in self.prevs:
			check &= (n.layer.out_size is not None)
		return check
	
	def computeSize(self):
		self.layer.in_size = [n.layer.out_size for n in self.prevs]
		self.layer.in_size_flatten = [np.prod(s) for s in self.layer.in_size]
		self.layer.out_size = self.layer.computeSize()
		self.layer.out_size_flatten = np.prod(self.layer.out_size)
		for n in self.nexts:
			# Solo se podra ejecutar si todas las dependencias han terminado de calcularse.
			if n.checkComputeSizeDependences():
				n.computeSize()

	""" 
		MISC
	"""
	# Compila la capa
	def compile(self):
		self.layer.compile()
		self.computeNumberOfBackwardNodes()

	# Determina que tipo de nodo es	
	def determineNode(self):
		if len(self.prevs) == 0 and len(self.nexts) == 0:
			self.type = Node.NOT_CONNECTED
			self.network.nodes_not_connected[self.name] = self
		elif len(self.prevs) == 0:
			self.type = Node.INPUT
			self.network.nodes_with_only_outputs[self.name] = self
		elif len(self.nexts) == 0:
			self.type = Node.OUTPUT
			self.network.nodes_with_only_inputs[self.name] = self
		else:
			self.type = Node.MIDDLE

	def fill(self, data):
		try:
			self.layer.fill(data)
		except AttributeError:
			raise Exceptions.LayerHasNoFillMethod("La capa " + self.node.name + "(" + type(self).__name__ + ") no puede llenarse.")
	
	def batchSize(self):
		return self.layer.batch_size

	def copy(self, copy_layer=False, name_prepend=''):
		c = self.__class__
		copy_node_instance = c.__new__(c)

		copy_node_instance.network = self.network
		copy_node_instance.name = name_prepend + self.name
		copy_node_instance.temp_forward_dependences = 0
		copy_node_instance.temp_backward_dependences = 0
		copy_node_instance.compute_forward_in_prediction = self.compute_forward_in_prediction
		copy_node_instance.compute_backward = self.compute_backward
		if copy_layer:
			copy_node_instance.layer = self.layer.copy(copy_node_instance)
		else:
			copy_node_instance.layer = self.layer 
		
		return copy_node_instance

	""" properties """
	@property
	def weights(self):
		return self.layer.weights