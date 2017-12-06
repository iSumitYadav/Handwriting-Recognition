import numpy as np
DEBUG_MODE = 0

class NeuralNetwork(object):
	"""
	Constructor
	Input:
		size_layer: list of number of neurons in each layer of the Neural Network
	Output:
		NeuralNetwork Object with 2 main members as weights and bias values for each layer neurons
	"""
	def __init__(self, size_layer):
		self.num_layers = len(size_layer)
		self.size = size_layer

		if DEBUG_MODE:
			# No bias for first(Input) layer
			self.bias = [np.round(np.random.randn(i, 1), 2) for i in size_layer[1:]]
			print("Bias Matrix for Network");print(self.bias)
			# No weight-Input for first(Input) layer and no weight-Output for last(Output) layer
			self.weight = [np.round(np.random.randn(j, i), 2) for i, j in zip(size_layer[:-1], size_layer[1:])]
			print("Weight Matrix for Network");print(self.weight)
		else:
			self.bias = [np.random.randn(i, 1) for i in size_layer[1:]]
			self.weight = [np.random.randn(j, i) for i, j in zip(size_layer[:-1], size_layer[1:])]

	def feedForward(self, a):
		if DEBUG_MODE:
			level = 1
		for b, w in zip(self.bias, self.weight):
			w_dot_a = np.dot(w, a)
			z = w_dot_a + b
			if DEBUG_MODE:
				print("Level " + str(level));print("weight");print(w);print("Input");print(a);print("Bias");print(b);print("weight.input");print(np.dot(w, a));print("z = weight.input + bias");print(np.dot(w, a) + b);
				a = np.round(sigmoid(np.dot(w, a) + b), 2)
				print("sigmoid(z)");print(a);print('='*30)
				level += 1
			else:
				a = sigmoid(np.dot(w, a) + b)
		return a

"""
sigmoid function
Input:
	z: vector or Numpy Array
Output:
	sigmoid value

Note: No need to implement sigmoid elementwise, Numpy applies the function sigmoid element-wise
"""
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

net = NeuralNetwork([1,2,3])
ouput = net.feedForward(np.array(1))

if DEBUG_MODE:
	print("Output of 3 Ouput Layer Neurons");print(ouput);
