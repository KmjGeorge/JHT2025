import torch
import torch.nn as nn
from src.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

@ARCH_REGISTRY.register()
class MLPClassifier(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, c_in, n_classes, hidden_dim):
        """
        bucket_size: int,
        n_hashes: int,
        """
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(c_in, hidden_dim), nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim, n_classes)


    def forward(self, x):
        x = self.fc2(self.fc1(x))    # (B, N, D) -> (B, N, C)
        x = F.log_softmax(x, dim=2)
        return x


