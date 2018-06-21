import h5py
import json
import numpy as np
import pydot
import copy
from .node import Node
from .layers.layer import Layer
from .optimizers import SGD
from .optimizers.optimizer import Optimizer
from .backend.exceptions import Exceptions

class Network(object):
	# Posibles estados de la red
	class STATUS(object):
		NOT_COMPILED, COMPILED, FORWARDED = range(3)
	
	def __init__(self):
		self.status = Network.STATUS.NOT_COMPILED
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

	def Node(self, name, layer, *layer_args, **layer_kargs):
		node = Node(self, name, layer, layer_args, layer_kargs)
		self.nodes[node.label] = node
		
		return node

	"""
		COMPILE:
			Genera e inicializa las estructuras de datos para ejecutar la red.
	"""
	def testHasInputs(self):
		if len(self.nodes_with_only_outputs) == 0:
			raise Exceptions.NotInputException("¡El grafo no tiene entrada!"+str([n for n in self.nodes_with_only_outputs]))
	
	def testHasOutputs(self):
		if len(self.nodes_with_only_inputs) == 0:
			raise Exceptions.NotOutputException("¡El grafo no tiene salida!"+str([n for n in self.nodes_with_only_inputs]))

	def testHasNotConnecteds(self):
		if len(self.nodes_not_connected) > 0:
			raise Exceptions.NotConnectedException("¡Existen nodos sin conectar! "+str([n for n in self.nodes_not_connected]))
	
	def testIsAcyclicGraph(self):
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

	def testAllInputsHaveData(self, only_no_targets=False):
		check = True
		for n in self.nodes_with_only_outputs.values():
			check &= (not only_no_targets and n.layer.has_data) or (n.compute_forward_in_prediction and n.layer.has_data)
			#(n.layer.has_data or (only_no_targets and not n.compute_forward_in_prediction))

	def computeSizes(self):
		for n in self.nodes.values():
			n.clearSizesResults()
		# Calculamos por cada nodo su tamaño
		for n in self.nodes_with_only_outputs.values():
			n.computeSize()

	def compile(self, losses, metrics = [], optimizer=SGD()):
		self.metrics = metrics
		# Calculamos los pesos de las losses
		# si la entrada ha sido una lista los pesos de cada loss son equiprobables.
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
		self.testHasInputs()
		self.testHasOutputs()
		self.testHasNotConnecteds()
		self.testIsAcyclicGraph()
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
		self.status = Network.STATUS.COMPILED

	def start(self, inputs, outputs):
		# Entradas y salidas de la red
		self.inputs = inputs
		
		# todos los nodos que solo tengan salidas y no esten en inputs deberan tener el compute_forward_in_prediction = False
		# Porque no participaran en la prediccion
		for n in self.nodes_with_only_outputs.values():
			if n not in self.inputs:
				n.compute_forward_in_prediction = False
		
		self.outputs = outputs

	def forward(self):
		# Recorremos los nodos que SOLO tienen salidas, es decir: Son la entrada de datos a la red.
		for n in self.nodes_with_only_outputs.values():
			# Solo se realiza el forward para elementos que participan en la preciccion.
			# Los tagets por ejemplo no se calculan.
			if not self.predict_flag or n.compute_forward_in_prediction:
				n.forward()
		# indicamos que hemos realizado ya el forward
		self.status = Network.STATUS.FORWARDED

	def backpropagation(self):
		# Si antes no se ha realizado un forward, debe hacerse.
		if self.status == Network.STATUS.COMPILED:
			self.forward()

		# Recorremos los nodos que SON LOSSES, es decir: Son la salida de datos a la red que determinan el error
		for n in self.losses:
			n.backpropagation(is_loss=True)
		self.status = Network.STATUS.COMPILED

	def train_batch(self, X, Y):
		for xk, xv in X.items():
			self.nodes_with_only_outputs[xk].fill(xv)
		for yk, yv in Y.items():
			self.nodes_with_only_outputs[yk].fill(yv)

		# Comprovamos que hemos llenado todos los inputs
		self.testAllInputsHaveData(only_no_targets=False)
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batchSize()
		
		self.backpropagation()

	def predict(self, X):
		self.predict_flag = True
		for xk, xv in X.items():
			if self.nodes_with_only_outputs[xk].compute_forward_in_prediction:
				self.nodes_with_only_outputs[xk].fill(xv)
		self.testAllInputsHaveData(only_no_targets=True)
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batchSize()
		self.forward()
		self.predict_flag = False
		return dict([(n.label, n.temp_forward_result) for n in self.outputs])

	def monitoring(self):
		print('Losses:', dict([(l.name, l.temp_forward_result) for l in self.losses]))
		print('Metrics:', dict([(m.name, m.temp_forward_result) for m in self.metrics]))

	def plot(self, filestr):
		graph = pydot.Dot(graph_type='digraph')
		graph.set_node_defaults(shape='none', fontname='Courier', fontsize='10')
		nodes = {}
		for i, n in enumerate(self.nodes.values()):
			nodes[n] = i
			graph.add_node(pydot.Node(i, label=
				u'<<table border="1" cellspacing="0"> \
					<tr><td border="1" sides="B" bgcolor="#dddddd"><font color="#d71414">'+n.name + ' (' + type(n.layer).__name__ + ')</font></td></tr> \
					<tr><td border="1" sides="B" style="dashed">in: ' + str(n.layer.in_size) + '</td></tr> \
					<tr><td border="0">out: ' + str(n.layer.out_size) + '</td></tr> \
				</table>>'))
		for n in self.nodes.values():
			for nn in n.nexts:
				graph.add_edge(pydot.Edge(nodes[n], nodes[nn]))
		graph.write_png(filestr)

	def get_weights(self, label):
		return self.nodes[label].weights

	def save(self, filename):
		if self.status == Network.STATUS.NOT_COMPILED:
			raise Exceptions.NotCompiledError("La red aun no ha sido compilada.")

		hf = h5py.File(filename, 'w')
		nodes_h5 = hf.create_group('nodes')

		# NODES
		nodes_json = {}
		nodes_id_dict = {}
		id_nodes_dict = {}
		for i, n in enumerate(self.nodes.values()):
			nodes_id_dict[i] = n
			id_nodes_dict[n] = i
			nodes_json[i] = n.save(nodes_h5.create_group(str(i)))

		# RELATIONS
		N = len(self.nodes)
		relations_json = {}
		for i, n in nodes_id_dict.items():
			relations_json[i] = []
			for nn in n.nexts:
				j = id_nodes_dict[nn]
				relations_json[i].append(j)

		# LOSSES
		losses_json = [id_nodes_dict[n] for n in self.losses]
		
		# METRICS
		metrics_json = [id_nodes_dict[n] for n in self.metrics]

		# OPTIMIZER
		optimizer_json = self.optimizer.save(hf.create_group("optimizer"))

		# JSON
		network_json = \
		{'status': self.status,
		'inputs': [id_nodes_dict[n] for n in self.inputs],
		'outputs': [id_nodes_dict[n] for n in self.outputs],
		'losses': losses_json,
		'metrics': metrics_json,
		'nodes': nodes_json,
		'relations': relations_json,
		'optimizer': optimizer_json}
		
		hf.create_dataset("data", data=json.dumps(network_json))
		hf.close()

	def load(self, filename):
		hf = h5py.File(filename, 'r')
		data = json.loads(hf['data'].value)

		self.status = data['status']

		# NODES
		id_nodes_dict = {}
		self.nodes = {}
		for i, n in data['nodes'].items():
			node = Node.load_static(self, n, hf['nodes'][i])
			id_nodes_dict[int(i)] = node
			self.nodes[node.label] = node

		# RELATIONS
		for i in data['relations'].keys():
			iint = int(i)
			for j in data['relations'][i]:
				id_nodes_dict[iint].addNext(id_nodes_dict[j])

		# Realizamos parte de la compilacion, esta no sobreescribe elementos cargados.
		for n in self.nodes.values():
			# limpiamos las dependencias
			n.clearForwardDependences()
			n.clearForwardResults()
			
			n.clearBackwardDependences()
			n.clearBackwardResults()

			# determinamos el tipo de nodo
			n.determineNode()

		for n in self.nodes.values(): # se dene ejecutar una vez compilados
			n.computeNumberOfBackwardNodes()

		# IN & OUT
		self.inputs = [id_nodes_dict[i] for i in data['inputs']]
		self.outputs = [id_nodes_dict[i] for i in data['outputs']]

		# LOSSES
		self.losses = [id_nodes_dict[i] for i in data['losses']]

		# METRICS
		self.metrics = [id_nodes_dict[i] for i in data['metrics']]


		# OPTIMIZER
		self.optimizer = Optimizer.load_static(data['optimizer'], hf['optimizer'])

		hf.close()


