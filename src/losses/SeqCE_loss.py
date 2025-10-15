import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from src.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SeqCELoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, logit_trains, label_trains):  # logit_trains (B, N, C)  N: seq length  C: classnum   # label_trains  (B, N)
        N = logit_trains.shape[1]
        ce_per_point = 0
        for i in range(N):
            ce_per_point += F.cross_entropy(logit_trains[:, i, :], label_trains[:, i], reduction=self.reduction)
        ce_for_seq = ce_per_point / N
        return ce_for_seq * self.loss_weight


if __name__ == '__main__':
    loss = CELossForSeq()
    logit_trains = torch.rand(size=(4, 1000, 5))
    label_trains = torch.randint(low=0, high=4, size=(4, 1000))
    print(loss(logit_trains, label_trains).item())
