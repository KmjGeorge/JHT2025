import numpy as np
import time
import matplotlib.pyplot as plt


def find_continuous_segments(labels):
    """
    在形状为(N, 1)的标签序列中找到所有标签连续的片段（长度至少为2）

    参数:
        labels: 形状为(N, 1)的标签序列，标签范围为0~k

    返回:
        字典:
            - 键: 标签种类 (int)
            - 值: 列表，每个元素是元组 (start, end) 表示连续片段的起止位置
    """
    # 将二维数组转换为一维数组
    labels_flat = labels.flatten()
    n = len(labels_flat)

    # 空序列处理
    if n == 0:
        return {}

    # 1. 找到所有变化点
    # 使用pad确保边界被正确处理
    padded = np.pad(labels_flat, (1, 1), 'constant', constant_values=(-1, -2))
    diff = np.diff(padded)
    change_points = np.where(diff != 0)[0]

    # 2. 计算所有片段的起始和结束位置
    starts = change_points[:-1]
    ends = change_points[1:] - 1

    # 3. 计算片段长度并过滤短片段
    lengths = ends - starts + 1
    valid_mask = lengths >= 2

    # 4. 获取有效片段的标签
    segment_labels = labels_flat[starts[valid_mask]]

    # 5. 获取有效片段的起止位置
    valid_starts = starts[valid_mask]
    valid_ends = ends[valid_mask]

    # 6. 按标签分组
    unique_labels = np.unique(segment_labels)
    segments_dict = {}

    for label in unique_labels:
        # 使用布尔索引高效选择当前标签的所有片段
        mask = segment_labels == label
        label_starts = valid_starts[mask]
        label_ends = valid_ends[mask]

        # 创建元组列表
        segments_dict[label] = list(zip(label_starts, label_ends))

    return segments_dict

def performance_test():
    # 创建大型测试数据
    sizes = [10 ** 5, 10 ** 6, 10 ** 7]
    loop_times = []
    vectorized_times = []

    for size in sizes:
        # 生成随机标签序列（包含大量短片段）
        labels = np.random.randint(0, 100, (size, 1))

        # 测试循环版本
        start = time.time()
        _ = find_continuous_segments_loop(labels)
        loop_time = time.time() - start
        loop_times.append(loop_time)

        # 测试向量化版本
        start = time.time()
        _ = find_continuous_segments(labels)
        vectorized_time = time.time() - start
        vectorized_times.append(vectorized_time)
        print(
            f"Size: {size:8d} | Loop: {loop_time:.5f}s | Vectorized: {vectorized_time:.5f}s | Speedup: {loop_time / vectorized_time:.1f}x")

    # 绘制性能对比图
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, loop_times, 'o-', label='Loop Implementation')
    plt.plot(sizes, vectorized_times, 's-', label='Vectorized Implementation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# 循环版本实现（用于性能对比）
def find_continuous_segments_loop(labels):
    labels_flat = labels.flatten()
    n = len(labels_flat)
    segments_dict = {}

    if n == 0:
        return segments_dict

    current_label = labels_flat[0]
    start_index = 0

    for i in range(1, n):
        if labels_flat[i] != current_label:
            if i - start_index > 1:
                if current_label not in segments_dict:
                    segments_dict[current_label] = []
                segments_dict[current_label].append((start_index, i - 1))

            current_label = labels_flat[i]
            start_index = i

    if n - start_index > 1:
        if current_label not in segments_dict:
            segments_dict[current_label] = []
        segments_dict[current_label].append((start_index, n - 1))

    return segments_dict


# 运行性能测试
if __name__ == "__main__":
    performance_test()