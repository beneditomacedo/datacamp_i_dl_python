# You're now going to practice calculating slopes. When plotting the
# mean-squared error loss function against predictions, the slope is 2 * x *
# (xb-y), or 2 * input_data * error. Note that x and b may have multiple
# numbers (x is a vector for each data point, and b is a vector). In this case,
# the output will also be a vector, which is exactly what you want.

# You're ready to write the code to calculate this slope while using a single
# data point. You'll use pre-defined weights called weights as well as data for
# a single point called input_data. The actual value of the target you want to
# predict is stored in target.

import numpy as np

# The data point you will make a prediction for
input_data = np.array([1, 2, 3])

# Sample weights
weights = np.array([0, 2, 1])

# The actual target value, used to calculate the error
target = 0

# Calculate the predictions: preds
preds = (input_data * weights).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)
