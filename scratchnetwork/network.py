import numpy as np
import pydot
import copy
from .node import Node
from .layers.layer import Layer
from .optimizers import SGD
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
		
		# optimizador
		self.optimizer = None

	def start(self, inputs, outputs):
		# Entradas y salidas de la red
		self.inputs = inputs
		
		# todos los nodos que solo tengan salidas y no esten en inputs deberan tener el compute_forward_in_prediction = True
		for n in self.nodes_with_only_outputs.values():
			if n not in self.inputs:
				n.compute_forward_in_prediction = False
		
		self.outputs = outputs

	def Node(self, name, layer, *layer_args):
		node = Node(self, name, layer, layer_args)
		self.nodes[name] = node
		
		return node

	"""
		COMPILE:
			Genera e inicializa las estructuras de datos para ejecutar la red.
	"""
	def testHasInputs(self):
		if len(self.nodes_with_only_outputs) == 0:
			raise Network.NotInputException("¡El grafo no tiene entrada!"+str([n for n in self.nodes_with_only_outputs]))
	
	def testHasOutputs(self):
		if len(self.nodes_with_only_inputs) == 0:
			raise Network.NotOutputException("¡El grafo no tiene salida!"+str([n for n in self.nodes_with_only_inputs]))

	def testHasNotConnecteds(self):
		if len(self.nodes_not_connected) > 0:
			raise Network.NotConnectedException("¡Existen nodos sin conectar! "+str([n for n in self.nodes_not_connected]))
	
	def testIsAcyclicGraph(self):
		# definimos una estructura para saber el camino recorrido
		path_nodes = []

		# funcion recursiva que nos permitira recorrer nodos
		def exploreNode(node):
			# Comprovamos que no este en el camino
			if node in path_nodes:
				raise Network.IsCyclicGraphException("¡El grafo es ciclico!")
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
			check &= (n.layer.has_data or (only_no_targets and not n.compute_forward_in_prediction))

	def computeSizes(self):
		for n in self.nodes.values():
			n.clearSizesResults()
		# Calculamos por cada nodo su tamaño
		for n in self.nodes_with_only_outputs.values():
			n.computeSize()

	def compile(self, losses, metrics = [], optimizer=SGD()):
		self.losses = losses
		self.metrics = metrics
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
		self.firstForward = True

		# Calculamos los tamaños, realmente aplicamos un "forward"
		# Las unicas capas que no tienen size de entrada son los inputs
		for n in self.nodes_with_only_outputs.values():
			n.computeSize()

		# Compilamos las capas
		for n in self.nodes.values():
			n.compile()

		# Calculamos los pesos de las losses
		# si la entrada ha sido una lista los pesos de cada loss son equiprobables.
		if isinstance(self.losses, list):
			v = 1./len(self.losses)
			self.weights_losses = {}
			for l in self.losses:
				self.weights_losses[l.layer] = v

		# si en cambio se ha introducido un diccionario del tipo {Loss1: 0.5, Loss2: 0.3, Loss3: 0.2}
		elif isinstance(self.losses, dict):
			self.weights_losses = self.losses
			self.losses = self.losses.keys()

		# cambiamos estado
		self.status = Network.STATUS.COMPILED



	def forward(self):
		# Recorremos los nodos que SOLO tienen salidas, es decir: Son la entrada de datos a la red.
		for n in self.nodes_with_only_outputs.values():
			# Solo se realiza el forward para elementos que participan en la preciccion.
			# Los tagets por ejemplo no se calculan.
			if not self.predict_flag or n.compute_forward_in_prediction:
				n.forward()
		# indicamos que hemos realizado ya el forward
		self.status = Network.STATUS.FORWARDED
		# indicamos que el primer forward ha sido realizado
		self.firstForward = False

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
			self.nodes_with_only_outputs[xk].fill(xv)
		self.testAllInputsHaveData(only_no_targets=True)
		# definimos el tipo tamano del batch
		self.batch_size = self.inputs[0].batchSize()
		self.forward()
		return dict([(n.name, n.temp_forward_result) for n in self.outputs])

		

	def monitoring(self):
		print('Losses:', dict([(l.name, l.temp_forward_result) for l in self.losses]))
		print('Metrics:', dict([(m.name, m.temp_forward_result) for m in self.metrics]))

	def plot(self, filestr):
		graph = pydot.Dot(graph_type='digraph')
		graph.set_node_defaults(shape='none', fontname='Courier', fontsize='10')
		nodes = {}
		for n in self.nodes.values():
			#graph.add_node(pydot.Node(n.name, label='{'+n.name + " (" + type(n.layer).__name__ + ")\\n|- In: " + str(n.layer.in_size)+ "\\n|- Out: " + str(n.layer.out_size) + '}'))
			graph.add_node(pydot.Node(n.name, label=
				u'<<table border="1" cellspacing="0"> \
					<tr><td border="1" sides="B" bgcolor="#dddddd"><font color="#d71414">'+n.name + ' (' + type(n.layer).__name__ + ')</font></td></tr> \
					<tr><td border="1" sides="B" style="dashed">in: ' + str(n.layer.in_size) + '</td></tr> \
					<tr><td border="0">out: ' + str(n.layer.out_size) + '</td></tr> \
				</table>>'))
		for n in self.nodes.values():
			for nn in n.nexts:
				graph.add_edge(pydot.Edge(n.name, nn.name))
		graph.write_png(filestr)

	def get_weights(self, name):
		return self.nodes[name].weights

	"""
		Excepciones

	"""
	class IsCyclicGraphException(Exception):
		pass
	class NotInputException(Exception):
		pass
	class NotOutputException(Exception):
		pass
	class NotFoundLayer(Exception):
		pass
	class InputShapeException(Exception):
		pass
	class InputNotFillException(Exception):
		pass
	class NumberInputsException(Exception):
		pass
	class NotConnectedException(Exception):
		pass
	class InconsistentWeightSize(Exception):
		pass
	class DifferentInputShape(Exception):
		pass
	class LayerHasNoFillMethod(Exception):
		pass