#
# Author:       cayscays
# Date:         December 2021
# Version:      1
# Description:  A neural network implemented from scratch
#

import random

import numpy as np


# A fully connected neural network
class NeuralNetwork:

    def __init__(self, input_size, hidden_layers_sizes, labels,
                 learning_rate, amount_of_epochs, batch_size, data, seed):
        random.seed(seed)
        self.all_layers_sizes = input_size + hidden_layers_sizes + labels
        self.learning_rate = learning_rate
        self.amount_of_epochs = amount_of_epochs
        self.batch_size = batch_size  # For future batch features. currently supports batch size of 1.

        self.errors = {'training': [], 'test': []}
        self.accuracy = {'training': [], 'test': []}
        self.epochs = []

        # Initiates random weights
        self.w = []
        for i in range(len(self.all_layers_sizes) - 1):
            self.w.append(np.random.rand(self.all_layers_sizes[i + 1], self.all_layers_sizes[i]))

        self.v = []
        for layer in self.all_layers_sizes:
            self.v.append(np.zeros(layer))

        self.delta = []
        for i in range(1, len(self.v)):
            self.delta.append(np.zeros(self.v[i].shape))

        # divide the data to test and training:
        random.shuffle(data)
        n = int(len(data) / 2)
        self.training_data = data[n:]
        self.test_data = data[:n]

    # Sigmoid
    # numpy array --> numpy array
    def sigmoid(self, vals):
        return 1 / (np.exp(-vals) + 1.0)

    # returns the label id
    # numpy array --> int
    def label_id(self, o):
        if (o[0] > 0.5):
            return 1
        else:
            return 0

    # updates all neuron's values (v)
    # returns the id of the predicted label
    # immidietly after the input layer no activation function
    # list --> int
    def forward_pass_single_input(self, single_input):
        self.v[0] = np.array(single_input)
        # forward pass
        for i in range(1, len(self.all_layers_sizes)):
            self.v[i] = self.sigmoid(self.w[i - 1] @ self.v[i - 1])
        return self.label_id(self.v[-1])

    # updates the weights according to backpropagation algorithm
    def backpropagation(self, currect_label):
        output = self.v[len(self.v) - 1]
        self.delta[len(self.delta) - 1] = (currect_label - output) * output * (1 - output)

        for l in range(len(self.delta) - 2, -1, -1):
            for i in range(len(self.delta[l])):
                temp = 0
                for j in range(len(self.delta[l + 1])):
                    temp += (self.w[l + 1][j][i] * self.delta[l + 1][j])
                self.delta[l][i] = self.v[l + 1][i] * (1 - self.v[l + 1][i]) * temp

        # update the weights
        for j in range(len(self.w)):
            for i in range(len(self.w[j])):
                # one line at a time:
                self.w[j][i] += self.learning_rate * self.v[j] * self.delta[j][i]

    # numpy, int, string ---> void
    def calculate_single_run_error(self, t, o):
        error = 0
        for i in range(len(o)):
            error += (t[i] - o[i]) ** 2
        return error

    # performs a run on all the data
    # updates the error, accurecy and weights
    def run_epoch(self):
        training_error = 0
        test_error = 0
        training_accuracy = 0
        test_accuracy = 0
        for i in range(len(self.training_data)):
            # 1 for correct, 0 for incorrect
            training_accuracy += 1 + self.forward_pass_single_input(self.training_data[i][0]) - \
                                 self.training_data[i][1][0]

            self.backpropagation(self.training_data[i][1])
            training_error += self.calculate_single_run_error(self.training_data[i][1], self.v[-1])

            # 1 for correct, 0 for incorrect
            test_accuracy += 1 + self.forward_pass_single_input(self.test_data[i][0]) - self.test_data[i][1][0]

            test_error += self.calculate_single_run_error(self.test_data[i][1], self.v[-1])
        test_error /= len(self.test_data)
        training_error /= len(self.training_data)
        self.errors['training'].append(training_error)
        self.errors['test'].append(test_error)

        test_accuracy /= len(self.test_data)
        training_accuracy /= len(self.training_data)
        test_accuracy *= 100
        training_accuracy *= 100
        self.accuracy['training'].append(training_accuracy)
        self.accuracy['test'].append(test_accuracy)

    # trains the network
    def train(self):
        for i in range(self.amount_of_epochs):
            self.run_epoch()
