import torch
import torch.nn.functional as F
from src.utils.registry import METRIC_REGISTRY




@METRIC_REGISTRY.register()
def calculate_nmi(pred_labels, true_labels):
    """
    计算归一化互信息 (Normalized Mutual Information, NMI)

    参数:
        true_labels : torch.Tensor, 真实标签
        pred_labels : torch.Tensor, 预测标签

    返回:
        nmi : float, 归一化互信息值 (范围在 [0, 1])
    """
    # 确保输入是整数类型的一维张量
    true_labels = true_labels.long()
    pred_labels = pred_labels.long()
    assert true_labels.shape == pred_labels.shape

    n = true_labels.shape[0]  # 样本数量

    # 计算真实标签和预测标签的类别
    classes_true = torch.unique(true_labels)
    classes_pred = torch.unique(pred_labels)

    # 特殊情况处理：如果所有样本属于同一类别
    if classes_true.shape[0] == 1 and classes_pred.shape[0] == 1:
        return 1.0
    elif classes_true.shape[0] == 1 or classes_pred.shape[0] == 1:
        return 0.0

    # 计算联合概率分布 P(i,j)
    contingency = torch.zeros((classes_true.shape[0], classes_pred.shape[0]), dtype=torch.float64)
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            contingency[i, j] = torch.sum((true_labels == c_true) & (pred_labels == c_pred))
    p_ij = contingency / n

    # 计算边缘概率 P(i) 和 P(j)
    p_i = torch.sum(p_ij, dim=1)
    p_j = torch.sum(p_ij, dim=0)

    # 计算互信息 MI
    mi = 0.0
    for i in range(p_i.shape[0]):
        for j in range(p_j.shape[0]):
            if p_ij[i, j] > 0:
                mi += p_ij[i, j] * torch.log(p_ij[i, j] / (p_i[i] * p_j[j]))

    # 计算熵 H(true) 和 H(pred)
    h_true = -torch.sum(p_i * torch.log(p_i))
    h_pred = -torch.sum(p_j * torch.log(p_j))

    # 计算归一化互信息 NMI (使用几何平均)
    nmi = mi / torch.sqrt(h_true * h_pred)
    return nmi
