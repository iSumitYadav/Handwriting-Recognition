import numpy as np
DEBUG_MODE = 1

class NeuralNetwork(object):
	def __init__(self, size_layer):
		self.num_layers = len(size_layer)
		self.size = size_layer

		if DEBUG_MODE:
			# No bias for first(Input) layer
			self.bias = [np.round(np.random.randn(i, 1), 2) for i in size_layer[1:]]
			print(self.bias)
			# No weight-Input for first(Input) layer and no weight-Output for last(Output) layer
			self.weight = [np.round(np.random.randn(j, i), 2) for i, j in zip(size_layer[:-1], size_layer[1:])]
			print(self.weight)
		else:
			self.bias = [np.random.randn(i, 1) for i in size_layer[1:]]
			self.weight = [np.random.randn(j, i) for i, j in zip(size_layer[:-1], size_layer[1:])]

net = NeuralNetwork([1,2,3])