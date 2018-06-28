import h5py
import numpy as np
from .layers.layer import Layer
from .layers.input import Input
from .backend.exceptions import Exceptions

import time
class Node(object):
	INPUT, OUTPUT, MIDDLE, NOT_CONNECTED = range(4)
	def __init__(self, network, name, layer, layer_args, layer_kargs):
		self.temp_forward_dependences = 0
		self.temp_backward_dependences = 0
		
		self.network = network


		if issubclass(layer, Layer):
			layer_args = tuple([self] + list(layer_args))
			self.layer = layer(*layer_args, **layer_kargs)
		else:
			raise Exceptions.NotFoundLayer("La capa que has introducido no existe.")

		if name is None:
			self.name = self.layer.__class__.__name__ + '_' + self.layer.LAYER_COUNTER
			self.label = self.name
		
		elif isinstance(name, (list, tuple)):
			self.name = name[1]
			self.label = name[0]
		else:
			self.name = name
			self.label = name
		self.type = None

		# Parametros pre-creacion de la capa
		self.compute_forward_in_prediction = True

		self.compute_backward = False
		# indicamos si produce o no dependencia tanto en el forward como el backward, las metricas no bloquean las dependencias ya que no participan en el juego solo evaluan.
		
		# relaciones que tiene el nodo
		self.prevs = []
		self.nexts = []

		# auxiliares
		self.number_backward_sum_nexts_nodes = None
		self.number_backward_any_prevs_nodes = None

	"""
		RELATIONS
	"""
	def addNext(self, node):
		self.nexts.append(node.start)
		node.prevs.append(self)


	def addPrev(self, node):
		self.prevs.append(node.end)
		node.nexts.append(self)

	def __call__(self, *vargs):
		if len(vargs) == 1:
			if isinstance(vargs[0], (list, tuple)):
				for n in vargs[0]:
					self.addPrev(n)
			else:
				self.addPrev(vargs[0])
		else:
			for n in vargs:
				self.addPrev(n)



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
		return self.layer.forward(inputs)
		
	def forward(self):
		# Si esta en modo no prediccion el forward se para en la Loss
		if self.network.predict_flag and not self.compute_forward_in_prediction:
			return

		#t = time.time()
		result = self.computeForward()
		#print('F: ',self.name, time.time() - t)
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

	def computeBackwardAndCorrect(self):
		# Obtenemos las derivadas de las proximas salidas
		doutput = self.temp_backward_result
		#print(self.compute_backward, self.name)
		unpack_derivatives = self.layer.derivatives(doutput)
		if isinstance(unpack_derivatives, (list, tuple)):
			backward, dweights = unpack_derivatives
		else:
			backward = unpack_derivatives
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
		
		#t = time.time()
		backward = self.computeBackwardAndCorrect()
		#print('B: ',self.name, time.time() - t)

		# si no tiene backward o no tiene nodos a los que enviar nada, el proceso se termina
		if not has_any_backward_to_compute or backward is None:
			return
		
		# si es una loss multiplicamos el peso asociado
		if is_loss:
			backward *= self.layer.weight

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
				#print(self.name, n.name, n.number_backward_sum_nexts_nodes, [m.compute_backward for m in n.nexts])
				if n.checkBackwardDependences():
					n.backpropagation()
					# una vez realizado el backward se vuelven a reinicializar las estructuras
					# en el backward debemos volver a poner a 0 tambien 
					n.clearBackwardDependences()

	def computeNumberOfBackwardNodes(self):
		# para no tener que calcularlo en cada iteracion, lo calculamos en la compilacion
		self.number_backward_sum_nexts_nodes = sum([int(n.compute_backward) for n in self.nexts])
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
		self.layer.in_size_flatten = [int(np.prod(s)) for s in self.layer.in_size]
		self.layer.out_size = [int(n) for n in self.layer.computeSize()]
		self.layer.out_size_flatten = int(np.prod(self.layer.out_size))
		for n in self.nexts:
			# Solo se podra ejecutar si todas las dependencias han terminado de calcularse.
			if n.checkComputeSizeDependences():
				n.computeSize()

	def hasToComputeBackward(self):
		self.compute_backward = True
		for n in self.prevs:
			if not n.compute_backward and n.layer.compute_backward:
				n.hasToComputeBackward()



	""" 
		MISC
	"""
	# Compila la capa
	def compile(self):
		self.layer.compile()

	# Determina que tipo de nodo es	
	def determineNode(self):
		if len(self.prevs) == 0 and len(self.nexts) == 0:
			self.type = Node.NOT_CONNECTED
			self.network.nodes_not_connected[self.label] = self
		elif len(self.prevs) == 0:
			self.type = Node.INPUT
			self.network.nodes_with_only_outputs[self.label] = self
		elif len(self.nexts) == 0:
			self.type = Node.OUTPUT
			self.network.nodes_with_only_inputs[self.label] = self
		else:
			self.type = Node.MIDDLE

	def fill(self, data):
		try:
			self.layer.fill(data)
		except AttributeError:
			raise Exceptions.LayerHasNoFillMethod("La capa " + self.name + "(" + type(self).__name__ + ") no puede llenarse.")
	
	def batchSize(self):
		return self.layer.batch_size

	def copy(self, copy_layer=False, name_prepend=None):
		c = self.__class__
		copy_node_instance = c.__new__(c)

		copy_node_instance.network = self.network

		if copy_layer:
			copy_node_instance.layer = self.layer.copy(copy_node_instance)
		else:
			copy_node_instance.layer = self.layer 
		
		if isinstance(name_prepend, str):
			copy_node_instance.name = name_prepend + self.name
			copy_node_instance.label = name_prepend + self.label

		else:
			copy_node_instance.name = self.name+'_'+copy_node_instance.layer.LAYER_COUNTER
			copy_node_instance.label = name_prepend + self.label+'_'+copy_node_instance.layer.LAYER_COUNTER

		copy_node_instance.temp_forward_dependences = 0
		copy_node_instance.temp_backward_dependences = 0
		copy_node_instance.compute_forward_in_prediction = self.compute_forward_in_prediction
		copy_node_instance.compute_backward = self.compute_backward
		
		return copy_node_instance

	""" properties """
	@property
	def weights(self):
		return self.layer.weights

	@property
	def end(self):
		return self

	@property
	def start(self):
		return self

	"""
		Save
	"""
	def save(self, h5_container):
		layer_json = self.layer.save(h5_container.create_group("layer"))
		
		return \
		{'label': self.label,
		'name': self.name,
		'compute_forward_in_prediction': self.compute_forward_in_prediction,
		'compute_backward': self.compute_backward,
		'layer': layer_json}

	@staticmethod
	def load_static(network, data, h5_container):
		obj = Node.__new__(Node)
		obj.label = data['label']
		obj.name = data['name']
		obj.compute_forward_in_prediction = data['compute_forward_in_prediction']
		obj.compute_backward = data['compute_backward']
		obj.layer = Layer.load_static(obj, data['layer'], h5_container['layer'])

		# constructor normal
		obj.temp_forward_dependences = 0
		obj.temp_backward_dependences = 0
		
		obj.network = network

		obj.prevs = []
		obj.nexts = []

		# auxiliares
		obj.number_backward_sum_nexts_nodes = None
		obj.number_backward_any_prevs_nodes = None
		return obj