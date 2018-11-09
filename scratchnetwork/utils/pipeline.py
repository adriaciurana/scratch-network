from ..node import Node
from ..layers.layer import Layer
from ..backend.exceptions import Exceptions
class Pipeline(object):
	def __init__(self, network, creator, name):
		self.network = network
		self.name = name
		self.subnet = Pipeline.SubNetwork()
		aux = creator(self.subnet)
		if len(aux) < 2 or len(aux) > 2:
			raise Exceptions.PipelineInputAndOutputFunction("Error, la funcion debe devolver la entrada y la salida.")

		if not isinstance(aux[0], Node) or not isinstance(aux[1], Node):
			raise Exceptions.PipelineInputAndOutputFunction("Error, la funcion debe devolver los nodos de la entrada y la salida.")

		self.input, self.output = aux
		self.num_copies = 0

	def copy(self, name=None, reuse=False):
		if name is None:
			name = self.name
		self.num_copies += 1

		relations = {}
		node_table = {}
		for n in self.subnet.nodes.values():
			if reuse:
				n.network = self.network
			node = n.copy(network=self.network, name_prepend=name + '/' + str(self.num_copies) + '/', copy_layer=not reuse, pipeline=self)
			if n == self.input:
				start = node
			if n == self.output:
				end = node
			relations[node] = n.nexts
			node_table[n] = node
			self.network.nodes[node.label] = node
		
		for n, rel in relations.items():
			for r in rel:
				n.addNext(node_table[r])

		return Pipeline.Instance(start, end)
	
	class Instance(object):
		def __init__(self, start, end):
			self.__start = start
			self.__end = end

		def __call__(self, *vargs):
			self.__start(*vargs)
			return self

		def addPrev(self):
			self.__start.addPrev(node)

		def addNext(self):
			self.__end.addNext(node)

		@property
		def end(self):
			return self.__end

		@property
		def start(self):
			return self.__start

		@property
		def nexts(self):
			return self.__end.nexts

		@property
		def prevs(self):
			return self.__start.nexts
		

	class SubNetwork(object):
		def __init__(self):
			self.nodes = {}

		def Node(self, layer, name, *layer_args, **layer_kwargs):
			node = Node(self, layer, name, layer_args, layer_kwargs)
			self.nodes[node.label] = node
			return node

		def add(self, layer, name, *layer_args, **layer_kwargs):
			return self.Node(layer, name, *layer_args, **layer_kwargs)

		def __call__(self, layer, name, *layer_args, **layer_kwargs):
			return self.Node(layer, name, *layer_args, **layer_kwargs)