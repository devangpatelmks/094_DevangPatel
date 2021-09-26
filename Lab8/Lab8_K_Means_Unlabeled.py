# CELL 0

# Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# CELL 1

X, _ = make_blobs(n_samples=100, centers=3, n_features=2, cluster_std=0.2, random_state=94)
# Scatter plot of the data points
# Basic Data Visualization
%matplotlib inline

plt.scatter(X[:, 0], X[:, -1])
plt.xlabel('X Coordinates')
plt.ylabel('Y Coordinates')
plt.show()

# CELL 2

# Using scikit-learn to perform K-Means clustering
# Specify the number of clusters (3) and fit the data X
kmeans = KMeans(n_clusters=3, random_state=94).fit(X)

# Visualize and evaluate the results
# Get the cluster centroids
print(kmeans.cluster_centers_)

# CELL 3
 
# Get the cluster labels
print(kmeans.labels_)

# CELL 4

# Plotting the cluster centers and the data points on a 2D plane
plt.scatter(X[:, 0], X[:, -1], c='orange', marker='*')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='green', marker='+')
plt.title('Data points and cluster centroids')
plt.show()

# CELL 5

# Calculate silhouette_score
print(silhouette_score(X, kmeans.labels_))

# CELL 6

# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=7)

# Instantiate the KElbowVisualizer with the number of clusters and the metric
visualizer = KElbowVisualizer(model, k=(2,6), metric='silhouette',timings=False)

# Fit the data and visualize
visualizer.fit(X)
visualizer.poof()