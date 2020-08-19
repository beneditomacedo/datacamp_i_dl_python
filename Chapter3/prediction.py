# The trained network from your previous coding exercise is now stored as 
# model. New data to make predictions is stored in a NumPy array as pred_data.
# Use model to make predictions on your new data.
# In this exercise, your predictions will be probabilities, which is the most
# common way for data scientists to communicate their predictions to 
# colleagues.

import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import ast


# Import df from file exported from datacamp with command
# df.to_dict(), copy and paste in vi file
#
file = open('data/titanic_dict.txt')
contents = file.read()
titanic_dict = ast.literal_eval(contents)
df = pd.DataFrame.from_dict(titanic_dict)

# Create predictors
# .as_matrix is deprecated. So the recommended is .to_numpy
# predictors = df.drop(['survived'], axis=1).as_matrix()
predictors = df.drop(['survived'], axis=1).to_numpy()

# Convert the target to categorical: target
target = to_categorical(df.survived)

# set n_cols
n_cols = 10

# Import pred_data from file exported from datacamp with command
# pred_data.tolist() and copy and paste into vi file
#
file = open('data/titanic_pred_data.txt')
contents = file.read()
pred_list = ast.literal_eval(contents)
pred_data = np.array(pred_list)

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# set to not use scientific notation
np. set_printoptions(suppress=True)

# print predicted_prob_true
print(predicted_prob_true)
