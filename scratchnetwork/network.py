import h5py, json, pydot, copy, time, math
import numpy as np
from .node import Node
from .layers.layer import Layer
from .layers.input import Input
from .layers.target import Target
from .layers.avoidfreeze import AvoidFreeze
from .utils.pipeline import Pipeline
from .optimizers.sgd import SGD
from .optimizers.optimizer import Optimizer
from .backend.exceptions import Exceptions
from .utils.prettyresults import PrettyResults
from .callbacks.prettymonitor import PrettyMonitor
from random import shuffle as random_shuffle

class Network(object):
	# Posibles estados de la red
	class STATUS(object):
		NOT_COMPILED, COMPILED, FREEZE = range(3)

	# Posibles conjuntos en los que se esta trabajando
	class SPLIT(object):
		TRAINING, VALIDATION = range(2)
	
	def __init__(self):
		self.status = self.STATUS.NOT_COMPILED
		self.predict_flag = False
		
		# Nodos
		self.nodes = {}
		self.nodes_with_only_inputs = {}
		self.nodes_with_only_outputs = {}
		self.nodes_not_connected = {}
		self.inputs = []
		self.outputs = []
		# ¿Que diferencia hay entre nodes_with_only_outputs y inputs,
		# y nodes_with_only_inputs y outputs?
		# La diferencia es que en la etapa de entrenamiento existen tanto capas de entrada como de salida que participan en el forward/backward nodes_with_only_outputs y nodes_with_only_inputs almacenan todos estos nodos
		# inputs y outputs solo almacenan los que seran en la prediccion las entradas y salidas de la red.
		
		# optimizador
		self.optimizer = None

		# losses y sus corresponientes pesos
		self.losses = []
		self.metrics = []

		self.batch_size = 0

	def Node(self, layer, name, *layer_args, **layer_kwargs):
		for n in self.nodes.values():
			if n.name == name:
				raise NameError('El nombre de esta capa ya existe')

		if layer is Pipeline:
			node = layer(self, layer_kwargs['creator'] if 'creator' in layer_kwargs else layer_args[0], name)
			return node
		
		else:
			node = Node(self, layer, name, layer_args, layer_kwargs)
			self.nodes[node.label] = node
			return node

	def add(self, layer, name, *layer_args, **layer_kwargs):
		return self.Node(layer, name, *layer_args, **layer_kwargs)

	def __call__(self, layer, name, *layer_args, **layer_kwargs):
		return self.Node(layer, name, *layer_args, **layer_kwargs)

	"""
		COMPILE:
			Genera e inicializa las estructuras de datos para ejecutar la red.
	"""
	def __testHasInputs(self):
		if len(self.nodes_with_only_outputs) == 0:
			raise Exceptions.NotInputException("¡El grafo no tiene entrada!"+str([n for n in self.nodes_with_only_outputs]))
	
	def __testHasOutputs(self):
		if len(self.nodes_with_only_inputs) == 0:
			raise Exceptions.NotOutputException("¡El grafo no tiene salida!"+str([n for n in self.nodes_with_only_inputs]))

	def __testHasNotConnecteds(self):
		if len(self.nodes_not_connected) > 0:
			raise Exceptions.NotConnectedException("¡Existen nodos sin conectar! "+str([n for n in self.nodes_not_connected]))
	
	def __testIsAcyclicGraph(self):
		# definimos una estructura para saber el camino recorrido
		path_nodes = []

		# funcion recursiva que nos permitira recorrer nodos
		def exploreNode(node):
			# Comprovamos que no este en el camino
			if node in path_nodes:
				raise Exceptions.IsCyclicGraphException("¡El grafo es ciclico!")
			# Lo añadimos al camino
			path_nodes.append(node)
			# Avanzamos al siguiente nodo
			for n in node.nexts:
				exploreNode(n)
			# Volvemos hacia atras y quitamos el nodo del camino
			path_nodes.pop()

		# Empezamos por todos los nodos que solo tienen salidas
		for n in self.nodes_with_only_outputs.values():
			# empezamos a recorrer
			exploreNode(n)

	def __testAllInputsHaveData(self, only_no_targets=False):
		check = True
		for n in self.nodes_with_only_outputs.values():
			check &= (not only_no_targets and n.layer.has_data) or (n.compute_forward_in_prediction and n.layer.has_data)
			#(n.layer.has_data or (only_no_targets and not n.compute_forward_in_prediction))

	def __testNetworkHasNotCompiled(self):
		if self.status == self.STATUS.NOT_COMPILED:
			raise Exceptions.NotCompiledError("La red aun no ha sido compilada.")

	def __testFreezeModel(self):
		if self.status == self.STATUS.FREEZE:
			raise Exceptions.NetworkAreFreezeModel("La red es un modelo congelado.")

	def __start(self):
		# iniciamos
		# todos los nodos que solo tengan salidas y no esten en inputs deberan tener el compute_forward_in_prediction = False
		# Porque no participaran en la prediccion
		for n in self.nodes_with_only_outputs.values():
			if n not in self.inputs:
				n.compute_forward_in_prediction = False

		# Comprovamos que 
		# Input no esta dentro de inputs
		"""for n in self.nodes.values():
			if isinstance(n.layer, Input) and n not in self.inputs:
				raise Exceptions.InputNotInInputs("El input "+ str(n.name) + " no esta dentro de inputs.")
			elif isinstance(n.layer, Target) and n in self.inputs:
				raise Exceptions.TargetInInputs("El target "+ str(n.name) + " no puede estar dentro de inputs.")"""


	def __compile(self):
		for n in self.nodes.values():
			# limpiamos las dependencias
			n.clearForwardDependences()
			n.clearForwardResults()
			
			n.clearBackwardDependences()
			n.clearBackwardResults()
			
			n.clearSizesResults()

			# determinamos el tipo de nodo
			n.determineNode()

		# Realizamos los tests
		self.__testHasInputs()
		self.__testHasOutputs()
		self.__testHasNotConnecteds()
		self.__testIsAcyclicGraph()
		
		# Habilitamos los tests en tiempo de ejecuccion
		self.predict_flag = False

		# Calculamos los tamaños, realmente aplicamos un "forward"
		# Las unicas capas que no tienen size de entrada son los inputs
		for n in self.nodes_with_only_outputs.values():
			n.computeSize()

		# Compilamos las capas
		for n in self.nodes.values():
			n.compile()

		for l in self.losses:
			l.hasToComputeBackward()
			
		for n in self.nodes.values(): # se dene ejecutar una vez compilados
			n.computeNumberOfBackwardNodes()

		# cambiamos estado
		self.status = self.STATUS.COMPILED

	def compile(self, inputs, outputs, losses, metrics = [], optimizer=SGD()):
		self.__testFreezeModel()

		if isinstance(metrics, Node):
			metrics = [metrics]
		self.metrics = metrics
		
		# Calculamos los pesos de las losses
		# si la entrada ha sido una lista los pesos de cada loss son equiprobables.
		if isinstance(losses, Node):
			losses = [losses]
		if isinstance(losses, list):
			v = 1./len(losses)
			for l in losses:
				l.layer.weight = v
			self.losses = losses

		# si en cambio se ha introducido un diccionario del tipo {Loss1: 0.5, Loss2: 0.3, Loss3: 0.2}
		elif isinstance(losses, dict):
			for k, v in losses.items():
				k.layer.weight = v
			self.losses = list(losses.keys())
		self.optimizer = optimizer
		
		# Compilamos la parte interna
		self.__compile()

		# Entradas y salidas de la red
		if isinstance(inputs, Node):
			inputs = [inputs]
		self.inputs = inputs

		if isinstance(outputs, Node):
			outputs = [outputs]
		self.outputs = outputs

		# Empezamos
		self.__start()

	def recompile(self):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()

		# Lo vuelve a inicializar todo
		self.__compile()

	"""
		FORWARD / BACKPROPAGATION & TRAINING
	"""
	def __forward(self):
		# Recorremos los nodos que SOLO tienen salidas, es decir: Son la entrada de datos a la red.
		for n in self.nodes_with_only_outputs.values():
			# Solo se realiza el forward para elementos que participan en la preciccion.
			# Los tagets por ejemplo no se calculan.
			if not self.predict_flag or n.compute_forward_in_prediction:
				n.forward()

	def __backpropagation(self):
		self.__forward()
			
		# Recorremos los nodos que SON LOSSES, es decir: Son la salida de datos a la red que determinan el error
		for n in self.losses:
			n.backpropagation(is_loss=True)

		self.optimizer.iteration()

	def train_batch(self, X, Y):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()

		for xk, xv in X.items():
			self.nodes_with_only_outputs[xk].fill(xv)
		for yk, yv in Y.items():
			self.nodes_with_only_outputs[yk].fill(yv)

		# Comprovamos que hemos llenado todos los inputs
		self.__testAllInputsHaveData(only_no_targets=False)
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batch_size
		
		self.__backpropagation()

	def __validate_batch(self, X, Y):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()

		for xk, xv in X.items():
			self.nodes_with_only_outputs[xk].fill(xv)
		for yk, yv in Y.items():
			self.nodes_with_only_outputs[yk].fill(yv)

		# Comprovamos que hemos llenado todos los inputs
		self.__testAllInputsHaveData(only_no_targets=False)
		
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batch_size
		self.__forward()

	def validate_batch(self, X, Y):
		self.__validate_batch(X, Y)
		return self.get_losses(), self.get_metrics

	def test_batch(self, X, Y):
		return self.validate_batch(X, Y)

	def fit(self, X, Y, epochs, batch_size, Xval=None, Yval=None, shuffle=True, iterations= {'training': None, 'validation': None}, callbacks=None):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()

		# Iniciamos los callbacks
		if callbacks is None:
			callbacks = [PrettyMonitor(PrettyMonitor.TRAINING, 5), PrettyMonitor(PrettyMonitor.VALIDATION)]
		for c in callbacks:
			c.init(self)

		# Redefinimos
		total_epochs = epochs
		enable_shuffle = shuffle

		# Flags para los parametros
		has_iterations = iterations is not None and iterations and isinstance(iterations, dict)
		enable_validation = Xval is not None and Yval is not None

		# Definimos el numero de iteraciones maximas por el training
		total_batchs = list(X.values())[0].shape[0]
		if has_iterations and 'training' in iterations and \
			iterations['training'] is not None:
			total_batchs = min(total_batchs, batch_size*iterations['training'])
		total_iterations = math.ceil(total_batchs / batch_size)

		# Definimos el numero de iteraciones maximas por el validation (si existe)
		if enable_validation:
			total_batchs_val = list(Xval.values())[0].shape[0]

			if has_iterations and 'validation' in iterations and \
				iterations['validation'] is not None:
				total_batchs_val = min(total_batchs_val, batch_size*iterations['validation'])
			total_iterations_val = math.ceil(total_batchs_val / batch_size)
		
		# Definimos los indices para realizar un shuffle de forma rapida
		indices = list(range(list(X.values())[0].shape[0]))
		if enable_validation:
			indices_val = list(range(list(Xval.values())[0].shape[0]))
		
		# Empezamos las epocas
		for epoch in range(total_epochs):
			# TRAINING
			#
			#
			# Aplicamos el shuffle
			if enable_shuffle:
				random_shuffle(indices)

			# Iniciamos la variables para training
			batch_index = 0
			iteration = 0
			# Empezamos las iteraciones
			for batch_index in range(0, total_batchs, batch_size):
				start_time = time.time()
				# Definimos los intervalos del batch
				start = batch_index
				end = min(batch_index + batch_size, total_batchs)

				# Cargamos el batch
				X_batch = dict((k, v[indices[start:end]]) for k, v in X.items())
				Y_batch = dict((k, v[indices[start:end]]) for k, v in Y.items())

				# Ejecutamos callbacks
				for c in callbacks:
					c.excecute_pre_batch(self.SPLIT.TRAINING, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs)
				
				# Entrenamos con ese batch
				self.train_batch(X_batch, Y_batch)
				
				# Ejecutamos callbacks
				end_time = time.time()
				for c in callbacks:
					c.excecute_post_batch(self.SPLIT.TRAINING, iteration, total_iterations, batch_index, total_batchs, batch_size, epoch, total_epochs, end_time - start_time)
				iteration += 1
				
			# VALIDATION
			#
			#
			# Comprobamos si hay que realizar validation
			if enable_validation:
				# Aplicamos el shuffle
				if enable_shuffle:
					random_shuffle(indices_val)
				
				# Iniciamos la variables para training
				batch_index = 0
				iteration = 0
				# Empezamos las iteraciones
				for batch_index in range(0, total_batchs_val, batch_size):
					start_time = time.time()
					# Definimos los intervalos del batch
					start = batch_index
					end = min(batch_index + batch_size, total_batchs)

					# Cargamos el batch
					X_batch = dict((k, v[indices[start:end]]) for k, v in X.items())
					Y_batch = dict((k, v[indices[start:end]]) for k, v in Y.items())

					# Ejecutamos callbacks
					for c in callbacks:
						c.excecute_pre_batch(self.SPLIT.VALIDATION, iteration, total_iterations_val, batch_index, total_batchs_val, batch_size, epoch, total_epochs)
					
					# Entrenamos con ese batch
					self.__validate_batch(X_batch, Y_batch)

					# Ejecutamos callbacks
					end_time = time.time()
					for c in callbacks:
						c.excecute_post_batch(self.SPLIT.VALIDATION, iteration, total_iterations_val, batch_index, total_batchs_val, batch_size, epoch, total_epochs, end_time - start_time)
					iteration += 1

	def predict(self, X):
		self.__testNetworkHasNotCompiled()

		self.predict_flag = True
		
		for xk, xv in X.items():
			if self.nodes_with_only_outputs[xk].compute_forward_in_prediction:
				self.nodes_with_only_outputs[xk].fill(xv)
		self.__testAllInputsHaveData(only_no_targets=True)
		
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batch_size
		self.__forward()
		
		self.predict_flag = False
		
		return dict([(n.label, n.temp_forward_result) for n in self.outputs])

	"""
		UTILES:
			
	"""
	def get_losses(self):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()
		return dict([(l.name, l.temp_forward_result) for l in self.losses])

	def get_metrics(self):
		self.__testFreezeModel()
		self.__testNetworkHasNotCompiled()
		return dict([(m.name, m.temp_forward_result) for m in self.metrics])
		
	def monitoring(self):
		print('Losses:', self.get_losses())
		print('Metrics:', self.get_metrics())

	def plot(self, filename):
		self.__testNetworkHasNotCompiled()

		graph = pydot.Dot(graph_type='digraph')
		graph.set_node_defaults(shape='none', fontname='Courier', fontsize='10')
		nodes = {}
		for i, n in enumerate(self.nodes.values()):
			reuse_html = ''
			if n.is_copied:
				if n.is_copied_reuse_layer:
					s = n.pipeline_name + ': shared parameters'
				else:
					s = n.pipeline_name + ': not shared parameters'
				reuse_html = '<tr><td border="1" sides="T" style="dashed">'+s+'</td></tr>'

			nodes[n] = i
			graph.add_node(pydot.Node(i, label=
				u'<<table border="1" cellspacing="0"> \
					<tr><td border="1" sides="B" bgcolor="#dddddd"><font color="#d71414">'+n.name + ' (' + type(n.layer).__name__ + ')</font></td></tr> \
					<tr><td border="1" sides="B" style="dashed">in: ' + str(n.layer.in_size) + '</td></tr> \
					<tr><td border="0">out: ' + str(n.layer.out_size) + '</td></tr> \
					'+ reuse_html +' \
				</table>>'))
		
		for n in self.nodes.values():
			for nn in n.nexts:
				graph.add_edge(pydot.Edge(nodes[n], nodes[nn]))
		
		graph.write_png(filename)

	def get_weights(self, label):
		self.__testNetworkHasNotCompiled()
		
		return self.nodes[label].weights

	def set_layer(self, label, *args, **kwargs):
		self.__testNetworkHasNotCompiled()

		return self.nodes[label].layer.set(*args, **kwargs)

	def save(self, filename, freeze=False):
		self.__testNetworkHasNotCompiled()

		# Definimos el freeze model
		freeze_model = freeze or self.status == self.STATUS.FREEZE
		
		# Creamos la libreria hf
		hf = h5py.File(filename, 'w')
		nodes_h5 = hf.create_group('nodes')

		# Nodos
		nodes_json = {}
		id_to_nodes = {}
		nodes_to_id = {}
		for i, n in enumerate(self.nodes.values()):
			if freeze_model and (isinstance(n.layer, AvoidFreeze) or all([isinstance(nn.layer, AvoidFreeze) for nn in n.prevs + n.nexts])):
				continue
			
			id_to_nodes[i] = n
			nodes_to_id[n] = i
			nodes_json[i] = n.save(nodes_h5.create_group(str(i)))

		# Relaciones (se almacenan como un diccionario)
		relations_json = {}
		for i, n in id_to_nodes.items():
			if freeze_model and isinstance(n.layer, AvoidFreeze):
				continue
			
			relations_json[i] = []
			for nn in n.nexts:
				if freeze_model and isinstance(nn.layer, AvoidFreeze):
					continue
			
				relations_json[i].append(nodes_to_id[nn])

		# Inputs
		inputs_json = []
		for n in self.inputs:
			if freeze_model and isinstance(n.layer, AvoidFreeze):
				continue
			
			inputs_json.append(nodes_to_id[n])

		# Outputs
		outputs_json = []
		for n in self.outputs:
			if freeze_model and isinstance(n.layer, AvoidFreeze):
				continue
			
			outputs_json.append(nodes_to_id[n])
		
		# Json
		network_json = {
		'status': self.status,
			'inputs': inputs_json,
			'outputs': outputs_json,
			'nodes': nodes_json,
			'relations': relations_json,
			'freeze_model': freeze_model
		}

		if not freeze_model:
			# Losses
			losses_json = [nodes_to_id[n] for n in self.losses]
			
			# Metricas
			metrics_json = [nodes_to_id[n] for n in self.metrics]
			
			# Optimizador
			optimizer_json = self.optimizer.save(hf.create_group("optimizer"))
			network_json['losses'] = losses_json
			network_json['metrics'] = metrics_json
			network_json['optimizer'] = optimizer_json		
		
		hf.create_dataset("data", data=json.dumps(network_json))
		hf.close()

	def load(self, filename):
		hf = h5py.File(filename, 'r')
		data = json.loads(hf['data'].value)

		self.status = data['status']

		# NODES
		id_to_nodes = {}
		self.nodes = {}
		for i, n in data['nodes'].items():
			node = Node.load_static(self, n, hf['nodes'][i])
			id_to_nodes[int(i)] = node
			self.nodes[node.label] = node

		# RELATIONS
		for i in data['relations'].keys():
			iint = int(i)
			for j in data['relations'][i]:
				id_to_nodes[iint].addNext(id_to_nodes[j])

		# Realizamos parte de la compilacion, esta no sobreescribe elementos cargados.
		for n in self.nodes.values():
			# limpiamos las dependencias
			n.clearForwardDependences()
			n.clearForwardResults()
			
			n.clearBackwardDependences()
			n.clearBackwardResults()

			# determinamos el tipo de nodo
			n.determineNode()

		# IN & OUT
		self.inputs = [id_to_nodes[i] for i in data['inputs']]
		self.outputs = [id_to_nodes[i] for i in data['outputs']]
		if data['freeze_model']:
			self.status = self.STATUS.FREEZE
		else:
			# LOSSES
			self.losses = [id_to_nodes[i] for i in data['losses']]

			# METRICS
			self.metrics = [id_to_nodes[i] for i in data['metrics']]

			# OPTIMIZER
			self.optimizer = Optimizer.load_static(data['optimizer'], hf['optimizer'])

			# Seteamos lo que necesitemos
			for l in self.losses:
				l.hasToComputeBackward()

			# Despues de cargar las losses
			for n in self.nodes.values(): # se debe ejecutar una vez compilados
				n.computeNumberOfBackwardNodes()

			for n in self.nodes_with_only_outputs.values():
				if n not in self.inputs:
					n.compute_forward_in_prediction = False

			self.status = self.STATUS.COMPILED

		# Cerramos hf
		hf.close()


