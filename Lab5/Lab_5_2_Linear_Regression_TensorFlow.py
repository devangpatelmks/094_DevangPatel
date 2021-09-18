# CELL 0

# Import necessary libraries
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import Normalizer
import sklearn.preprocessing

# CELL 1

# Linear Regression Model Using Tensor Flow

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70], [73, 67, 43],
                   [91, 88, 64], [87, 134, 58], [102, 43, 37],
                   [69, 96, 70], [73, 67, 43], [91, 88, 64],
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
# Targets (apples)
targets = np.array([[56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103], 
                    [56], [81], [119], [22], [103]], dtype='float32')
n = len(inputs) # Number of data points

# CELL 2

from sklearn.model_selection import train_test_split
main_data = pd.DataFrame(inputs, columns = ['temp', 'rainfall', 'humidity'])
Y_rows = pd.DataFrame(targets, columns = ['apples'])

X_train, X_test, Y_train, Y_test = train_test_split(main_data, Y_rows, test_size = 0.25, random_state = 94)

# CELL 3

test_results = {}

norm_X_test = np.linalg.norm(X_test['rainfall'])
norm_X_train = np.linalg.norm(X_train['rainfall'])
norm_Y_train = np.linalg.norm(Y_train['apples'])

normal_array_X_test = X_test['rainfall']/norm_X_test
normal_array_X_train = X_train['rainfall']/norm_X_train
normal_array_Y_train = Y_train['apples']/norm_Y_train

print(normal_array_X_train)
print(normal_array_X_test)
print(normal_array_Y_train)

# CELL 4

normalizer = preprocessing.Normalization(axis = -1)
normalizer.adapt(np.array(X_train))
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units = 1)
])
print(linear_model.predict(X_train[ : 9]))

# CELL 5

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate = 0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    X_train, Y_train, 
    epochs = 1000,
    # Suppress logging
    verbose = 0,
    # Calculate validation results on 25% of the training data
    validation_split = 0.25)

test_results['linear_model'] = linear_model.evaluate(
    X_test, Y_test, verbose = 0)

test_results['linear_model']