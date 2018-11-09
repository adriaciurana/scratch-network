class Exceptions:
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
	class NotCompiledError(Exception):
		pass
	class NetworkAreFreezeModel(Exception):
		pass
	class InputNotInInputs(Exception):
		pass
	class TargetInInputs(Exception):
		pass

	# Pipeline
	class PipelineInputAndOutputFunction(Exception):
		pass
	class PipelineInputAndOutputFunction(Exception):
		pass

	# Operation
	class OperationNotExist(Exception):
		pass

	# Custom cannot compile without functions
	class CustomNotCompile(Exception):
		pass