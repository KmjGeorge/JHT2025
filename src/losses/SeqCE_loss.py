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

    def forward(self, prob_trains,
                label_trains):  # prob_trains (B, N, C)  N: seq length  C: classnum   # label_trains  (B, N)
        B, N, C = prob_trains.shape
        label_trains = label_trains.long()
        return F.cross_entropy(prob_trains.view(B*N, C), label_trains.view(B*N), reduction=self.reduction)


if __name__ == '__main__':
    loss = SeqCELoss()
    B = 4
    N = 1000
    C = 5
    logit_trains = F.softmax(torch.rand(size=(B, N, C)), dim=2)
    print(logit_trains)
    label_trains = torch.randint(low=0, high=C - 1, size=(B, N))
    print(loss(logit_trains, label_trains).item())

    logit_trains = logit_trains.view(B*N, C)
    label_trains = label_trains.view(B*N)
    print(logit_trains.shape, label_trains.shape)
    print(F.cross_entropy(logit_trains, label_trains, reduction='mean').item())
