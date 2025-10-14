from copy import deepcopy

from src.utils.registry import METRIC_REGISTRY
from ari import calculate_ari
from ac import calculate_ac
from nmi import calculate_nmi
from deinterleaving_score import calculate_deinterleaving_score
__all__ = ['calculate_ac', 'calculate_ari', 'calculate_nmi', 'deinterleaving_score']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
