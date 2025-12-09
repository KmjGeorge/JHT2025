import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import matplotlib

class HierarchicalDBSCAN:
    def __init__(self, window_size, step_size, eps1=0.5, min_samples1=5, eps2=0.5, min_samples2=5):
        """
        初始化分层DBSCAN聚类器

        参数:
        window_size (int): 窗口长度w
        step_size (int): 步长s
        eps1 (float): 第一层DBSCAN的邻域半径
        min_samples1 (int): 第一层DBSCAN的最小样本数
        eps2 (float): 第二层DBSCAN的邻域半径
        min_samples2 (int): 第二层DBSCAN的最小样本数
        """
        self.window_size = window_size
        self.step_size = step_size
        self.eps1 = eps1
        self.min_samples1 = min_samples1
        self.eps2 = eps2
        self.min_samples2 = min_samples2
        self.segment_centers = []
        self.segment_labels_map = {}

    def _split_into_segments(self, X):
        """将数据按窗口和步长切分成段"""
        segments = []
        segment_indices = []

        n_samples = X.shape[0]
        start = 0

        while start < n_samples:
            end = min(start + self.window_size, n_samples)
            segment = X[start:end]
            segments.append(segment)
            segment_indices.append((start, end))
            start += self.step_size

        return segments, segment_indices

    def _compute_cluster_centers(self, segment, labels):
        """计算每个聚类的中心点"""
        centers = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:  # 噪声点，跳过
                continue
            cluster_points = segment[labels == label]
            center = np.mean(cluster_points, axis=0)
            centers.append(center)

        return centers

    def fit_predict(self, X):
        """
        执行分层DBSCAN聚类

        参数:
        X (numpy.ndarray): 形状为(N, D)的输入数据

        返回:
        numpy.ndarray: 形状为(N,)的最终聚类标签
        """
        # 1. 数据切分
        segments, segment_indices = self._split_into_segments(X)
        all_centers = []
        center_to_segment_map = []  # 记录每个中心点对应的原始段信息

        print(f"将数据切分成 {len(segments)} 个段")

        # 2. 对每个段进行DBSCAN聚类
        for i, segment in enumerate(segments):
            if len(segment) < 2:  # DBSCAN需要至少2个样本
                continue

            # 标准化当前段的数据
            scaler = StandardScaler()
            segment_scaled = scaler.fit_transform(segment)

            # 第一层DBSCAN聚类
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dbscan1 = DBSCAN(eps=self.eps1, min_samples=self.min_samples1)
                segment_labels = dbscan1.fit_predict(segment_scaled)

            # 计算聚类中心
            centers = self._compute_cluster_centers(segment, segment_labels)

            # 记录中心点和对应的段信息
            for j, center in enumerate(centers):
                all_centers.append(center)
                center_to_segment_map.append({
                    'segment_idx': i,
                    'center_idx': j,
                    'original_indices': segment_indices[i],
                    'segment_labels': segment_labels
                })

            print(f"段 {i}: 包含 {len(segment)} 个点, 生成 {len(centers)} 个聚类中心")

        if not all_centers:
            print("警告: 未生成任何聚类中心")
            return np.array([-1] * len(X))  # 所有点标记为噪声

        all_centers = np.array(all_centers)
        print(f"总共生成 {len(all_centers)} 个聚类中心")

        # 3. 对所有聚类中心进行第二次DBSCAN聚类
        if len(all_centers) > 1:
            # 标准化聚类中心
            scaler_centers = StandardScaler()
            centers_scaled = scaler_centers.fit_transform(all_centers)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dbscan2 = DBSCAN(eps=self.eps2, min_samples=self.min_samples2)
                center_labels = dbscan2.fit_predict(centers_scaled)
        else:
            center_labels = np.array([0])  # 只有一个中心点

        # 4. 将新的标签分配给原始数据
        final_labels = np.full(len(X), -1)  # 初始化为-1（噪声）
        current_label = 0

        # 创建从中心标签到最终标签的映射
        label_mapping = {}
        for center_label in np.unique(center_labels):
            if center_label == -1:
                continue
            label_mapping[center_label] = current_label
            current_label += 1

        # 为每个数据点分配标签
        for center_idx, center_info in enumerate(center_to_segment_map):
            seg_idx = center_info['segment_idx']
            start, end = center_info['original_indices']
            segment_labels = center_info['segment_labels']

            # 获取该中心对应的最终标签
            center_label = center_labels[center_idx]
            if center_label == -1:
                final_label = -1  # 噪声
            else:
                final_label = label_mapping[center_label]

            # 找到该中心对应的原始数据点并分配标签
            corresponding_cluster_label = None
            unique_labels = np.unique(segment_labels)
            cluster_count = 0

            for label in unique_labels:
                if label != -1:
                    if cluster_count == center_info['center_idx']:
                        corresponding_cluster_label = label
                        break
                    cluster_count += 1

            if corresponding_cluster_label is not None:
                # 为属于该聚类的所有点分配最终标签
                segment_mask = (segment_labels == corresponding_cluster_label)
                global_indices = np.arange(start, end)[segment_mask]
                final_labels[global_indices] = final_label

        # 处理未被分配标签的点（将它们标记为噪声）
        noise_mask = final_labels == -1
        if np.any(noise_mask):
            print(f"有 {np.sum(noise_mask)} 个点被标记为噪声")

        return final_labels


def demo_hierarchical_dbscan():
    """演示如何使用分层DBSCAN"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    # 创建三个不同的聚类
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(70, n_features))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(70, n_features))
    cluster3 = np.random.normal(loc=[10, 0], scale=0.5, size=(60, n_features))
    cluster4 = np.random.normal(loc=[20, 0], scale=0.5, size=(100, n_features))

    X = np.vstack([cluster1, cluster2, cluster3, cluster4])

    print(f"生成数据形状: {X.shape}")

    # 创建分层DBSCAN实例
    hdbscan = HierarchicalDBSCAN(
        window_size=50,  # 窗口长度
        step_size=25,  # 步长
        eps1=0.3,  # 第一层DBSCAN参数
        min_samples1=3,
        eps2=0.5,  # 第二层DBSCAN参数
        min_samples2=2
    )

    # 执行聚类
    labels = hdbscan.fit_predict(X)

    # 输出结果
    unique_labels = np.unique(labels)
    print(f"\n最终聚类结果:")
    print(f"发现 {len(unique_labels) - (1 if -1 in unique_labels else 0)} 个聚类")

    for label in unique_labels:
        if label == -1:
            print(f"噪声点: {np.sum(labels == label)} 个")
        else:
            print(f"聚类 {label}: {np.sum(labels == label)} 个点")

    return X, labels


def sliding_window_normalization(features_list, window_size=100, overlap=0.5):
    """
    滑动窗口标准化，保持时序局部特性

    参数:
    features_list: 多段特征列表
    window_size: 窗口大小
    overlap: 窗口重叠比例
    """
    normalized_features = []

    for features in features_list:
        L, C = features.shape
        step = int(window_size * (1 - overlap))

        # 创建标准化后的特征数组
        normalized = np.zeros_like(features)
        counts = np.zeros(L)  # 记录每个点被标准化了多少次

        # 滑动窗口处理
        for start in range(0, L, step):
            end = min(start + window_size, L)
            if end - start < 2:  # 窗口太小跳过
                continue

            window_features = features[start:end]

            # 窗口内标准化
            window_mean = np.mean(window_features, axis=0)
            window_std = np.std(window_features, axis=0)
            window_std[window_std == 0] = 1.0  # 避免除零

            normalized_window = (window_features - window_mean) / window_std

            # 累加到结果中
            normalized[start:end] += normalized_window
            counts[start:end] += 1

        # 平均处理
        mask = counts > 0
        normalized[mask] /= counts[mask, np.newaxis]

        # 对未被处理的点进行全局标准化
        if not np.all(mask):
            global_mean = np.mean(features[mask], axis=0)
            global_std = np.std(features[mask], axis=0)
            global_std[global_std == 0] = 1.0

            normalized[~mask] = (features[~mask] - global_mean) / global_std

        normalized_features.append(normalized)
    return normalized_features


if __name__ == "__main__":
    # 运行演示
    X, labels = demo_hierarchical_dbscan()

    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.6)
    plt.title('原始数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')

    plt.subplot(1, 2, 2)
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:
            color = 'black'
        else:
            color = colors[i]
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=f'聚类{label}', alpha=0.6)

    plt.title('分层DBSCAN聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.tight_layout()
    plt.show()
