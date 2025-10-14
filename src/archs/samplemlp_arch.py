import torch
import torch.nn as nn
import torch.nn.functional as F
from reformer_pytorch import LSHSelfAttention
import math
from src.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SimpleMLP(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, seq_len, c_in, c_out, d_model):
        """
        bucket_size: int,
        n_hashes: int,
        """
        super(SimpleMLP, self).__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Sequential(nn.Linear(c_in, d_model), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(d_model, d_model * 3), nn.ReLU())
        self.fc3 = nn.Linear(d_model * 3, d_model * 3)
        self.fc4 = nn.Linear(d_model * 3, c_out)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.fc4(self.drop2(self.fc3(self.drop1(self.fc2(self.fc1(x))))))
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = SimpleMLP(seq_len=3000, c_in=5, c_out=32, d_model=256).cuda()
    # x = torch.randn(8, 100, 3)
    # y = model(x)
    # print(y.shape)
    # for i in range(1000):
    #     x = torch.randn(4, 3000, 5).cuda()
    #     y = model(x)
    summary(model, input_size=(3000, 5))
