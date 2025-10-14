import torch
import torch.nn.functional as F
from src.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_ari(pred_labels, true_labels):
    """
    计算调整兰德指数 (Adjusted Rand Index)
    :param true_labels: 真实标签，形状为 [n_samples]
    :param pred_labels: 预测标签，形状为 [n_samples]
    :return: ARI 值
    """
    pred_labels = pred_labels.long()
    true_labels = true_labels.long()

    n_samples = true_labels.shape[0]

    # 创建真实标签和预测标签的独热编码
    true_onehot = F.one_hot(true_labels).float()
    pred_onehot = F.one_hot(pred_labels).float()

    # 计算混淆矩阵
    contingency = torch.mm(true_onehot.t(), pred_onehot)

    # 计算行和列的和
    a_sum = contingency.sum(dim=1)
    b_sum = contingency.sum(dim=0)

    # 计算组合数
    comb_a = torch.sum(a_sum * (a_sum - 1)) / 2
    comb_b = torch.sum(b_sum * (b_sum - 1)) / 2
    comb_samples = n_samples * (n_samples - 1) / 2

    # 计算配对数
    nij = 0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            nij += contingency[i, j] * (contingency[i, j] - 1) / 2

    # 计算期望值
    expected_index = comb_a * comb_b / comb_samples
    max_index = 0.5 * (comb_a + comb_b)

    # 计算调整兰德指数
    ari = (nij - expected_index) / (max_index - expected_index + 1e-10)
    return ari
