import numpy as np


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustment = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    print('Random synaptic weights: ')
    print(neural_network.synaptic_weights)

    print('## STARTING TRAINING SESSION ##')
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]
                                ])
    print('> Training input: ')
    print(training_inputs)

    training_outputs = np.array([[0, 1, 1, 0]]).T
    print('> Training output: ')
    print(training_outputs)

    neural_network.train(training_inputs, training_outputs, 100000)

    print('> Synaptic weights after training:')
    print(neural_network.synaptic_weights)

    while True:
        a = float(input('Input 1: '))
        b = float(input('Input 2: '))
        c = float(input('Input 3: '))

        print('New input: ', a, b, c)
        print('Output:')
        print(neural_network.think(np.array([[a, b, c]])))
