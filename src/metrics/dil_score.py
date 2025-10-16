from src.utils.registry import METRIC_REGISTRY
# from scipy.optimize import linear_sum_assignment
import numpy as np

@METRIC_REGISTRY.register()
def calculate_dil_score(pred_labels, true_labels):    # (B, N)     (B, N)
   B, N = pred_labels.shape
   r_fby = 0
   r_mby = 0
   for pred_label, true_label in zip(pred_labels, true_labels):
      C = [] # 分选结果的簇
      T = [] # 真实标签的簇
      pred_unique = np.unique(pred_label)
      true_unique = np.unique(true_label)

      true_cls_num = len(true_unique)
      for label in pred_unique:
         C.append(pred_labels[pred_labels == label])
      for label in true_unique:
         T.append(true_labels[true_labels == label])
      addtional_batches, missing_batches = calculate_additional_and_missing_batches(T, C)
      # print(addtional_batches, missing_batches)
      r_fby += addtional_batches / true_cls_num
      r_mby += missing_batches / true_cls_num
   r_fby /= B
   r_mby /= B
   # print(r_fby, r_mby)
   score = 0
   if r_mby == 0:
      score += 40
   elif 0 < r_mby <= 0.02:
      score += 32
   elif 0.02 < r_mby <= 0.05:
      score += 24
   elif 0.05 < r_mby <= 0.1:
      score += 16
   elif 0.1 < r_mby <= 0.15:
      score += 8
   else:
      score += 0

   if r_fby == 0:
      score += 40
   elif 0 < r_fby <= 0.02:
      score += 32
   elif 0.02 < r_fby <= 0.05:
      score += 24
   elif 0.05 < r_fby <= 0.08:
      score += 16
   elif 0.08 < r_fby <= 0.1:
      score += 8
   else:
      score += 0

   return score


def calculate_additional_and_missing_batches(T, C):
   """
   计算增批数量和漏批数量

   参数:
   T (list of sets): 真实类别列表，每个元素是一个脉冲集合（Ti）
   C (list of sets): 分选结果类别列表，每个元素是一个脉冲集合（Cj）

   返回:
   additional_batches (int): 增批数量
   missing_batches (int): 漏批数量
   """
   n = len(T)  # 真实类别数
   m = len(C)  # 分选结果类别数
   # 用于记录哪些Cj被匹配过（至少被一个Ti选为最佳匹配且匹配度≥30%）
   matched_C = set()

   # 遍历每个真实类别Ti
   for i, Ti in enumerate(T):
      best_match_score = 0.0
      best_match_index = -1

      # 遍历所有分选结果Cj，寻找Ti的最佳匹配
      for j, Cj in enumerate(C):
         intersection = np.intersect1d(Ti, Cj)  # 计算交集
         score = len(intersection) / len(Ti) if len(Ti) > 0 else 0.0

         # 更新最佳匹配
         if score > best_match_score:
            best_match_score = score
            best_match_index = j

      # 如果最佳匹配的匹配度≥30%，则记录该Cj被匹配
      if best_match_score >= 0.3 and best_match_index != -1:
         matched_C.add(best_match_index)

   # 计算增批数量：没有被任何Ti匹配的Cj数量
   additional_batches = m - len(matched_C)

   # 计算漏批数量：真实类别数 - 被匹配的Cj数量
   missing_batches = n - len(matched_C)

   return additional_batches, missing_batches



if __name__ == '__main__':
   pred_labels = np.array([[1, 2, 0, 1, 2], [0, 1, 2, 3, 0]])
   true_labels = np.array([[1, 1, 2, 3, 4], [3, 2, 1, 0, 0]])
   score = calculate_dil_score(pred_labels, true_labels)
   print(score)