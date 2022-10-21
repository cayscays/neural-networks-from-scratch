#
# Author:       cayscays
# Date:         December 2021
# Version:      1
# Description:  Train a neural network to classify oscillators on Conway's game of life.
#               The network's architecture:
#                   * Input layer made of 49 neurons.
#                   * 2 mid-layers of size 7.
#                   * Output layer made of a single neuron.
#

import matplotlib.pyplot as plt
import pandas as pd

import dataset
import neuralnetwork as nn

SEED = 10

# Network's architecture:
INPUT_SIZE = [49]
HIDDEN_LAYERS_SIZES = [7, 7]
LABELS = [1]

# Optimization parameters:
LEARNING_RATE = 0.5
amount_of_epochs = 30
batch_size = 1

# Initiate and train the neural network
nn1 = nn.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYERS_SIZES, LABELS, LEARNING_RATE, amount_of_epochs, batch_size,
                       dataset.data, SEED)
nn1.train()

# Initiate epochs for the x axes of the graphs
epochs = []
for i in range(amount_of_epochs):
    epochs.append(i)

# Plot error and accuracy graphs:
fig, axs = plt.subplots(1, 2)

error_graph = pd.DataFrame(nn1.errors, epochs)
error_graph.plot(title="Error", kind='line', xlabel='Number of epochs', ax=axs[0], ylabel="Error")

accuracy_graph = pd.DataFrame(nn1.accuracy, epochs)
accuracy_graph.plot(title="Accuracy", kind='line', xlabel='Number of epochs', ax=axs[1], ylabel="Accuracy")
plt.tight_layout()
plt.show()

print("The test accuracy is " + str(nn1.accuracy['test'][-1]) + "%")
