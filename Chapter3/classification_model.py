# You'll now create a classification model using the titanic dataset, which has
# been pre-loaded into a DataFrame called df. You'll take information about the
# passengers and predict which ones survived.
# The predictive variables are stored in a NumPy array predictors. The target to
# predict is in df.survived, though you'll have to manipulate it for keras. The
# number of predictive features is stored in n_cols.
# Here, you'll use the 'sgd' optimizer, which stands for Stochastic Gradient
# Descent. You'll learn more about this in the next chapter!

import pandas as pd
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

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)

# Print model summary
print(model.summary())
