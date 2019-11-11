%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])

df.head()

# Getting the values and plotting it
# You can use any two features from the four variables
feature1 = df['Sepal Length'].values
feature2 = df['Petal Length'].values
X = np.array(list(zip(feature1, feature2)))
plt.scatter(feature1, feature2, c='black', s=7)

# Euclidean Distance Caculator
def distance_measure(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
np.random.seed(17)
# Random Centroids in X Direction
x_centroid = np.random.randint(np.mean(X), np.max(X), size=k)
# Random Centroid in Y Direction
y_centroid = np.random.randint(np.mean(X), np.max(X), size=k)
centroid = np.array(list(zip(x_centroid, y_centroid)), dtype=np.float32)
print(centroid)

# Plotting Centroids
plt.scatter(feature1, feature2, c='#050505', s=7)
plt.scatter(x_centroid, y_centroid, marker='*', s=200, c='g')

# storing the previous centroid values
pre_centroid = np.zeros(centroid.shape)
# defining cluster labels
label_clusters = np.zeros(len(X))
# distance computation for error measurement
err = distance_measure(centroid, pre_centroid, None)

# reduce the error
while err != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        dist_ = distance_measure(X[i], centroid)
        c = np.argmin(dist_)
        label_clusters[i] = c
    # Storing the old centroid values
    pre_centroid = deepcopy(centroid)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if label_clusters[j] == i]
        centroid[i] = np.mean(points, axis=0)
    err = distance_measure(centroid, pre_centroid, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if label_clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroid[:, 0], centroid[:, 1], marker='*', s=200, c='#050505')
plt.title('K_means_scratch')
# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input df
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print(centroid) # From Scratch
print(centroids) # From sci-kit learn
fig, ax = plt.subplots()
for i in range(k):
        points__ = np.array([X[j] for j in range(len(X)) if labels[j] == i])
        ax.scatter(points__[:, 0], points__[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.title('K_means_Sklearn')



