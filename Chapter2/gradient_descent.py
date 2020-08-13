import numpy as np


input_data = np.array([3, 4])
weights = np.array([1, 2])
target = 6
learning_rate = 0.01
preds = (input_data * weights).sum()
error = preds - target
print(f'Original error is {error}')

gradient = 2 * input_data * error
print(f'Gradient is {gradient}')

weights_updated = weights - learning_rate * gradient
print(f'Weights {weights}')
print(f'Weights updated {weights_updated}')

preds_updated = (input_data * weights_updated).sum()
print(f'Original preds is {preds}')
print(f'Updated preds is {preds_updated}')

error_updated = preds_updated - target
print(f'Original error is {error}')
print(f'Updated error is {error_updated}')

gradient_2 = 2 * input_data * error_updated
print(f'Second gradient is {gradient_2}')

weights_2 = weights_updated - learning_rate * gradient_2
print(f'Weights {weights}')
print(f'Weights updated {weights_updated}')
print(f'Second Weights {weights_2}')

preds_2 = (input_data * weights_2).sum()
print(f'Original preds is {preds}')
print(f'Updated preds is {preds_updated}')
print(f'Second preds is {preds_2}')

error_2 = preds_2 - target
print(f'Original error is {error}')
print(f'Updated error is {error_updated}')
print(f'Second error is {error_2}')