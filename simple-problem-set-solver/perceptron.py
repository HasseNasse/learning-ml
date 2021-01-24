import numpy as np


# Activation / Normalization function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Activation / Normalization function derivative
def sigmoid_derivative(x):
    return x * (1 - x)


# Output calculation, used inside each neuron
def calculate_output(input, weights):
    return sigmoid(np.dot(input, weights))


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

# Set random weights
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('> Random starting weights: ')
print(synaptic_weights)

# Training the model
print('### TRAINING THE NEURAL NETWORK ### ')
for iteration in range(100_000):
    input_layer = training_inputs
    outputs = calculate_output(input_layer, synaptic_weights)
    error = training_outputs - outputs

    # Correction
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('> Synaptic weights after training:')
print(synaptic_weights)

print('> Outputs after training:')
print(outputs)

print('## NN EVALUATION ##')
evaluation_input = np.array([[1, 0, 0],
                             [0, 0, 0],
                             [1, 1, 0]])
evaluation_output = calculate_output(evaluation_input, synaptic_weights)

print('> NN Evaluation input:')
print(evaluation_input)
print('> NN Evaluation result:')
print(evaluation_output)
