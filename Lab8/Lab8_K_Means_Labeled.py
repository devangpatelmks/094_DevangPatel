# CELL 0

# Import libraries
# Using scikit-learn to perform K-Means clustering
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CELL 1

digits = load_digits()
print(digits.data.shape)

# CELL 2

kmeans = KMeans(n_clusters = 10, random_state = 94)
prediction = kmeans.fit_predict(digits.data)
print(prediction)
print(kmeans.cluster_centers_.shape)

# CELL 3
 
# Scatter plot of the data points

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)

for axi, center in zip(ax.flat, centers):
  axi.set(xticks=[], yticks=[])
  axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

# CELL 4

labels = np.zeros_like(prediction)
for i in range(10):
  mask = (prediction == i)
  labels[mask] = mode(digits.target[mask])[0]

accuracy_score(digits.target, labels)

mat = confusion_matrix(digits.target, labels)

ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                 xticklabels=digits.target_names, yticklabels=digits.target_names)
ax.set_ylim(10.0,0)

plt.xlabel('Predicted label')