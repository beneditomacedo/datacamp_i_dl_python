# The Rectified Linear Activation Function
#
# As Dan explained to you in the video, an "activation function" is a function
# applied at each node. It converts the node's input into some output.
#
# The rectified linear activation function (called ReLU) has been shown to lead
# to very high-performance networks. This function takes a single number as an
# input, returning 0 if the input is negative, and the input if the input is
# positive.
#
# Here are some examples: relu(3) = 3 relu(-3) = 0
#
# ----------------------------------------------------------------------------
# Great work! You predicted 52 transactions. Without this activation function,
# you would have predicted a negative number! The real power of activation
# functions will come soon when you start tuning model weights.

import numpy as np


def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    # Return the value just calculated
    return(output)


input_data = np.array([3, 5])
weights = {'node_0': np.array([2, 4]),
           'node_1': np.array([4, -5]),
           'output': np.array([2, 7])}
node_0_value = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_value)

node_1_value = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_value)

hidden_layer_values = np.array([node_0_output, node_1_output])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)
