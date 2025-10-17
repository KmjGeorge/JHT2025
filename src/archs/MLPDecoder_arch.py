import torch
import torch.nn as nn
from src.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

@ARCH_REGISTRY.register()
class MLPDecoder(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, c_in, c_out, hidden_dim):
        """
        bucket_size: int,
        n_hashes: int,
        """
        super(MLPDecoder, self).__init__()
        self.norm = nn.LayerNorm(c_in)
        self.fc1 = nn.Sequential(nn.Linear(c_in, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.out_proj = nn.Linear(hidden_dim, c_out)


    def forward(self, x):
        x = self.norm(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        x = self.out_proj(x)
        return x


