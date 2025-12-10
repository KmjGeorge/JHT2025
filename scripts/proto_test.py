import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

proto = torch.rand(size=(640, 128)).numpy()
proto_2 = torch.empty(640, 128)
nn.init.orthogonal_(proto_2)
tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=250)
proto_tsne1 = tsne.fit_transform(proto)
proto_tsne2 = tsne.fit_transform(proto_2)
plt.scatter(proto[:, 0], proto[:, 1], c='blue', s=1)
plt.show()
plt.scatter(proto_2[:, 0], proto_2[:, 1], c='blue', s=1)
plt.show()
plt.scatter(proto_tsne1[:, 0], proto_tsne1[:, 1], c='blue', s=1)
plt.show()
plt.scatter(proto_tsne2[:, 0], proto_tsne2[:, 1], c='blue', s=1)
plt.show()