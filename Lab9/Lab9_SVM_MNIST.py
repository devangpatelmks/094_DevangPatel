# CELL 0

# Import necessary libraries

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# CELL 1

# SVM classifier on MNIST dataset, compare the preformance of linear, polynomial and RBF kernels.

digits = load_digits()
digits.data.shape

# CELL 2

print(digits.target)

# CELL 3

# Split the data

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3 , random_state = 94)

# CELL 4

# Linear SVM classifier

lsc = svm.SVC(kernel = 'linear')
lsc.fit(X_train, y_train)

# CELL 5

lin_y_pred = lsc.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, lin_y_pred))

# CELL 6

# RBF SVM Classifier¶
rbf_svm = svm.SVC(kernel = 'rbf')
rbf_svm.fit(X_train, y_train)

# CELL 7

rbf_y_pred = rbf_svm.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, rbf_y_pred))

# CELL 8

# Polynominal SVM Classifier
p_svm = svm.SVC(kernel = 'poly')
p_svm.fit(X_train, y_train)

# CELL 9

poly_y_pred = p_svm.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, poly_y_pred))