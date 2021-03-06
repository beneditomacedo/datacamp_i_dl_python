# Now you'll get to work with your first model in Keras, and will immediately
# be able to run more complex neural network models on larger datasets
# compared to the first two chapters.
# To start, you'll take the skeleton of a neural network and add a hidden layer
# and an output layer. You'll then fit that model and see Keras do the
# optimization so your model continually gets better.
# As a start, you'll predict workers wages based on characteristics like their
# industry, education and level of experience. You can find the dataset in a
# pandas dataframe called df. For convenience, everything in df except for the
# target has been converted to a NumPy matrix called predictors. The target,
# For all exercises in this chapter, we've imported the Sequential model
# constructor, the Dense layer constructor, and pandas.

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

#
# TODO if the names parameter is used during genfromtxt import the nd.array
# shape will be wrong
#
predictors = np.genfromtxt("data/hourly_wages.csv", delimiter=',',
                           usecols=np.r_[1:10], skip_header=1)

target = np.genfromtxt("data/hourly_wages.csv", delimiter=',',
                       usecols=(0), skip_header=1)

# Checking the predictors shape
assert len(predictors.shape) == 2, "look in np.gemfromtxt import comment"
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation="relu", input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation="relu"))

# Add the output layer
model.add(Dense(1))

# Print model summary
print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
model.fit(predictors, target)

# Save the model
model.save('data/class_model.h5')
