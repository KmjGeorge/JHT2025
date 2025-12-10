import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch

root = r''
all_labels = []
for filename in tqdm(os.listdir(root)):
    t = torch.load(os.path.join(root, filename))
    label = t[:, -1].long()
    all_labels.append(label)

all_labels = torch.cat(all_labels, dim=0).numpy()
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.title('Label Histogram')
plt.hist(all_labels, bins=[i+1 for i in range(354)])
label_unique, counts = np.unique(all_labels, return_counts=True)
weights = len(all_labels) / (counts * len(label_unique))
weights = weights * 10 / max(weights)
plt.subplot(122)
plt.title('Focal Weight')
plt.plot(weights)
plt.tight_layout()
plt.savefig('label_hist_and_weight.png')

np.save(r'save_focal_weights4k.npy', weights)
with open(r'save_focal_weights4k.txt', 'w') as f:
    f.write(str(weights))