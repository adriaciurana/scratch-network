from ..node import Node
from ..layers.layer import Layer
class Pipeline(object):
	def __init__(self, network, name, creator):
		self.network = network
		self.name = name
		self.subnet = Pipeline.SubNetwork()
		aux = creator(self.subnet)
		if len(aux) < 2 or len(aux) > 2:
			print("Error, la funcion debe devolver la entrada y la salida.")
			exit()

		if not isinstance(aux[0], Node) or not isinstance(aux[1], Node):
			print("Error, la funcion debe devolver los nodos de la entrada y la salida.")
			exit()

		self.input, self.output = aux
		self.copies = 0

	def copy(self, name=None, reuse=False):
		if name is None:
			name = self.name
		self.copies += 1

		relations = {}
		node_table = {}
		for n in self.subnet.nodes.values():
			if reuse:
				n.network = self.network
			node = n.copy(copy_layer=not reuse, name_prepend=name + '_' + str(self.copies) + "_", network=self.network)
			node.reuse = reuse
			node.pipeline_name = self.name
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

		def Node(self, name, layer, *layer_args, **layer_kargs):
			node = Node(self, name, layer, layer_args, layer_kargs)
			self.nodes[node.label] = node
			return node