import torch
import torch.nn.functional as F
from src.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_ac(pred_labels, true_labels):
    """
    计算准确率 (Accuracy)
    :param true_labels: 真实标签，形状为 [n_samples]
    :param pred_labels: 预测标签，形状为 [n_samples]
    :return: AC 值
    """
    # 使用匈牙利算法找到最佳标签映射
    pred_labels = pred_labels.long()
    true_labels = true_labels.long()
    n_classes = max(true_labels.max(), pred_labels.max()) + 1
    one_hot_true = F.one_hot(true_labels, n_classes).float()
    one_hot_pred = F.one_hot(pred_labels, n_classes).float()

    # 构建代价矩阵
    cost_matrix = torch.mm(one_hot_true.t(), one_hot_pred)
    cost_matrix = -cost_matrix.cpu().numpy()  # 转换为负值用于最小化问题

    # 使用匈牙利算法求解
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 创建映射后的预测标签
    mapping = torch.zeros(n_classes, dtype=torch.long)
    mapping[row_ind] = torch.tensor(col_ind)
    mapped_pred = mapping[pred_labels]

    # 计算准确率
    correct = torch.sum(mapped_pred == true_labels)
    return correct.float() / true_labels.shape[0]

