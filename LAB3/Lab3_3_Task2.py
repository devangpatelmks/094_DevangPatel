# CELL 0
# Import necessary libraries

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# CELL 1
# Load the wine dataset and split the data

dataset = load_wine()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size = 0.20, random_state = 94)

# CELL 2
# GaussianNB

gnb = GaussianNB()

# Train the model
gnb.fit(X_train, Y_train)

# Complete training
Y_predicted = gnb.predict(X_test)

print(f"Accuracy:- {metrics.accuracy_score(Y_test, Y_predicted)}")