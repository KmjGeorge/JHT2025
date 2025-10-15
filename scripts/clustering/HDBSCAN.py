import hdbscan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
data, labels_gt = make_blobs(n_samples=5000, n_features=32, centers=2)
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)
print(cluster_labels)
plt.figure(figsize=(8, 10))
plt.subplot(211)
plt.title('HDBSCAN Results')
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)
plt.subplot(212)
plt.title('GT')
plt.scatter(data[:, 0], data[:, 1], c=labels_gt)
plt.show()
