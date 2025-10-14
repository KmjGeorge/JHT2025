from copy import deepcopy

from src.utils.registry import METRIC_REGISTRY
from src.metrics.ari import calculate_ari
from src.metrics.ac import calculate_ac
from src.metrics.nmi import calculate_nmi
from src.metrics.deinterleaving_score import calculate_deinterleaving_score
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
