# CELL 0

import numpy as np
import seaborn as sns 
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# CELL 1

dataset=datasets.load_breast_cancer()
dataset

print(dataset.data.shape)
print(dataset.target.shape)

# CELL 2

print(dataset.feature_names)
print(dataset.target_names)

# CELL 3

# 0 for benign and 1 for malignant

plt.scatter(dataset.data[:, 0], dataset.target, c='red', marker='*')
plt.xlabel('Features')
plt.ylabel('Type of Cancer')
plt.show()

# CELL 4

kmeans = KMeans(n_clusters = 7, random_state = 94)
prediction = kmeans.fit_predict(dataset.data)
print(prediction)

# CELL 5

kmeans.cluster_centers_.shape
print(kmeans.cluster_centers_)

# CELL 6

plt.scatter(dataset.data[:, 0], dataset.target, c='orange', marker='*')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='green', marker='+')
plt.title('Data points and cluster centroids')
plt.show()

# CELL 7

from scipy.stats import mode
labels = np.zeros_like(prediction)
for i in range(10):
  mask = (prediction == i)
  labels[mask] = mode(dataset.target[mask])[0]

accuracy_score(dataset.target, labels)

# CELL 8

mat = confusion_matrix(dataset.target, labels)
ax = sns.heatmap(mat.T, square=True, annot=True, cbar=False, xticklabels=dataset.target_names, yticklabels=dataset.target_names)

plt.xlabel('True label')
plt.ylabel('Predicted label')