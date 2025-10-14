import torch
import torch.nn.functional as F
from src.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_deinterleaving_score(pred_labels, true_labels):
    pass
