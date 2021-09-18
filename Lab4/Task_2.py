# CELL 0
# Import scikit-learn dataset library

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load dataset
digit = datasets.load_digits()

# CELL 1
# Print the names of the features

# Print the label type of digits
print("\n class: \n",digit.target_names)

# Print data(feature)shape
print( "\n",digit.data.shape)

# CELL 2
# Import the necessary module

from sklearn.model_selection import train_test_split

# Split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(digit.data, digit.target, test_size = 0.20, random_state = 94)

# CELL 3
# Create a Decision Tree Classifier (using Gini)

cli = DecisionTreeClassifier(criterion = 'gini', max_leaf_nodes=94)
cli.fit(data_train, target_train)

# Train the model using the training sets

# CELL 4
# Predict the classes of test data
prediction = cli.predict(data_test)
# Print(test_pred.dtype)
prediction.dtype

# CELL 5
# Model Accuracy, how often is the classifier correct?
from sklearn import metrics 
print("Accuracy: ", metrics.accuracy_score(target_test, prediction))

# CELL 6
precision = precision_score(target_test, prediction, average=None)
recall = recall_score(target_test, prediction, average=None)
print('precision: \n {}'.format(precision))
print('\n')
print('recall: \n {}'.format(recall))