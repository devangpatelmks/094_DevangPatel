# CELL 0
# Import scikit-learn dataset library

import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Load dataset
iris = datasets.load_iris()

# CELL 1
# Print the names of the features
print("Features: ", iris.feature_names)

# Print the label type of wine(class_0, class_1, class_2)
print("Labels: ", iris.target_names)

# Print data(feature)shape
print("\nData shape: ", iris.data.shape)
# Print data(target)shape
print("\nTarget shape: ", iris.target.shape)

print("\nData type: ", type(iris.data))

newdata = iris.data[50:, :] # Consider 50 to 100 as new data
newtarget = iris.target[50:]

# Print data(feature)shape
print("\nNew Data shape: ", newdata.shape)
# Print data(target)shape
print("\nNew Target shape: ", newtarget.shape)

# CELL 2
# Import the necessary module
from sklearn.model_selection import train_test_split

# Split data set into train and test sets
# 30% for testing and 70% for training
# random_state is the seed for randomization
data_train, data_test, target_train, target_test = train_test_split(newdata, 
newtarget, test_size = 0.30, random_state = 5)

# CELL 3
import numpy as np
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(data_train, target_train)

# Predict the response for test dataset
target_pred = gnb.predict(data_test)

# CELL 4
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(target_test, target_pred))

# CELL 5
# Import confusion_matrix from scikit-learn metrics module for confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(target_test, target_pred)

# CELL 6
# Calculate precision score and recall score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(target_test, target_pred)
recall = recall_score(target_test, target_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))