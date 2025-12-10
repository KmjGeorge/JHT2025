import numpy as np


def segment_and_process(data, window_size, stride, process_func):
    """
    将数据分段处理并融合重叠部分

    参数:
    data -- 输入数据，形状为(N, D)的NumPy数组
    window_size -- 窗口长度W
    stride -- 窗口移动步长S
    process_func -- 处理函数，输入(W_i, D)数组，返回相同形状的处理结果

    返回:
    result -- 最终处理结果，形状为(N, D)
    """
    n = data.shape[0]

    # 验证参数有效性
    if window_size <= 0:
        raise ValueError("窗口大小必须大于0")
    if stride <= 0:
        raise ValueError("步长必须大于0")

    # 初始化结果数组和计数数组
    result = np.zeros_like(data)
    count = np.zeros(n, dtype=int)

    # 计算窗口数量
    num_windows = (n - window_size) // stride + 1
    remainder = n - (num_windows - 1) * stride - window_size

    # 如果剩余部分可以构成一个窗口（即使小于W）
    if remainder > 0:
        num_windows += 1

    # 处理每个窗口
    for i in range(num_windows):
        # 计算当前窗口的起始和结束索引
        start = i * stride
        end = min(start + window_size, n)

        # 获取当前窗口数据
        window_data = data[start:end]

        # 应用处理函数
        processed_window = process_func(window_data)

        # 确保处理结果形状正确
        if processed_window.shape != window_data.shape:
            raise ValueError(f"处理函数返回形状错误: 期望 {window_data.shape}, 实际 {processed_window.shape}")

        # 将处理结果累加到最终结果
        result[start:end] += processed_window

        # 更新计数
        count[start:end] += 1

    # 处理计数为0的位置（理论上不应该出现）
    if np.any(count == 0):
        zero_indices = np.where(count == 0)[0]
        raise RuntimeError(f"错误：位置 {zero_indices} 没有被任何窗口覆盖")

    # 计算平均值（重叠部分取平均）
    result /= count[:, np.newaxis]

    return result


# 示例处理函数
def example_process_func(window):
    """示例处理函数：简单地将窗口数据乘以2"""
    return (window + np.random.rand() / 100).round(3)


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    N = 10  # 数据点数量
    D = 2  # 数据维度
    data = np.array([[i,i,i,i,i] for i in range(10)], dtype=float)

    print("原始数据:")
    print(data)

    # 设置参数
    W = 5  # 窗口大小
    S = 3  # 步长

    # 执行分段处理
    result = segment_and_process(data, W, S, example_process_func)

    print("\n处理结果:")
    print(result)

    # 验证非重叠部分
    print("\n验证非重叠部分:")
    print("位置0-2的原始数据:", data[0:3])
    print("位置0-2的处理结果:", result[0:3])

    # 验证重叠部分
    print("\n验证重叠部分:")
    # 位置3-4在两个窗口中都有出现
    print("位置3-4的原始数据:", data[3:5])
    print("位置3-4的处理结果:", result[3:5])
    print("位置3-4的平均值:", (data[3:5] + data[3:5]) / 2)

    # 验证最后一个窗口
    print("\n验证最后一个窗口:")
    print("位置6-9的原始数据:", data[6:10])
    print("位置6-9的处理结果:", result[6:10])