import torch
import torch.nn.functional as F
from src.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_nmi(pred_labels, true_labels):
    """
    计算归一化互信息 (Normalized Mutual Information)
    :param true_labels: 真实标签，形状为 [n_samples]
    :param pred_labels: 预测标签，形状为 [n_samples]
    :return: NMI 值
    """
    pred_labels = pred_labels.long()
    true_labels = true_labels.long()

    n_samples = true_labels.shape[0]

    # 创建真实标签和预测标签的独热编码
    true_onehot = F.one_hot(true_labels).float()
    pred_onehot = F.one_hot(pred_labels).float()

    # 计算联合概率分布
    joint = torch.mm(true_onehot.t(), pred_onehot) / n_samples

    # 计算边缘概率
    true_marginal = true_onehot.mean(dim=0)
    pred_marginal = pred_onehot.mean(dim=0)

    # 计算互信息
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:
                mi += joint[i, j] * torch.log(joint[i, j] / (true_marginal[i] * pred_marginal[j]))

    # 计算真实标签和预测标签的熵
    h_true = -torch.sum(true_marginal * torch.log(true_marginal + 1e-10))
    h_pred = -torch.sum(pred_marginal * torch.log(pred_marginal + 1e-10))

    # 计算归一化互信息
    return mi / torch.sqrt(h_true * h_pred)
