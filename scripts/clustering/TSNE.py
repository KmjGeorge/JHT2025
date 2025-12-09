from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import random
np.random.seed(42)
random.seed(42)
data, labels_gt = make_blobs(n_samples=300, n_features=32, centers=15)
tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=250)

out_tsne = tsne.fit_transform(data)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('TSNE')
plt.scatter(out_tsne[:, 0], out_tsne[:, 1], c=labels_gt)
plt.subplot(122)
plt.title('Data')
plt.scatter(data[:, 0], data[:, 1], c=labels_gt)
plt.show()