# In this exercise, you'll write code to do forward propagation (prediction) for
# your first neural network.
# Each data point is a customer. The first input is how many accounts they have,
# and the second input is how many children they have. The model will predict
# how many transactions the user makes in the next year. You will use this data
# throughout the first 2 chapters of this course.

# The input data has been pre-loaded as input_data, and the weights are
# available in a dictionary called weights. The array of weights for the first
# node in the hidden layer are in weights['node_0'], and the array of weights
# for the second node in the hidden layer are in weights['node_1'].

# The weights feeding into the output node are available in weights['output'].

# ----------------------------------------------------------------------------
# Wonderful work! It looks like the network generated a prediction of -39.

import numpy as np

input_data = np.array([3, 5])
weights = {'node_0': np.array([2, 4]),
           'node_1': np.array([4, -5]),
           'output': np.array([2, 7])}
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()
hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)
