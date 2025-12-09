import torch
from sklearn.manifold import TSNE
# from src.archs.samplemlp_arch import SimpleMLP
from src.archs.Flowformer_arch import Flowformer

import os
import numpy as np
from tqdm import tqdm
import hdbscan
import matplotlib.pyplot as plt
from src.archs.Reformer_arch import Reformer
from src.archs.samplemlp_arch import SimpleMLP
from src.data.data_util import normalize_zscore, normalize_minmax, draw_pdwtrain_with_label, PDWTrain, \
    read_pdw_new_recog
import torch.nn.functional as F

def t_sne(feature):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=250)
    feature_tsne = tsne.fit_transform(feature)
    return feature_tsne


def get_model(model_name, weight_path):
    if model_name == 'SimpleMLP':
        model = SimpleMLP(seq_len=3000,
                          c_in=5,
                          c_out=32,
                          d_model=256).cuda()
    elif model_name == 'Flowformer':
        model = Flowformer(seq_len=230000,
                           enc_in=5,
                           d_model=128,
                           c_out=32,
                           n_heads=8,
                           d_ff=512,
                           activation='relu',
                           dropout=0.1,
                           e_layers=8,
                           pe_mode='RoPE')
    elif model_name == 'Reformer':
        model = Reformer(
            seq_len=2000,
            enc_in=5,
            d_model=128,
            c_out=32,
            n_heads=8,
            d_ff=512,
            activation='relu',
            dropout=0.1,
            e_layers=8,
            bucket_size=8,
            n_hashes=8)

    model.load_state_dict(torch.load(weight_path)['params'])
    model = model.cuda()
    return model



def segment_inference(data, window_size, stride, model):
    """
    将数据分段处理并融合重叠部分
    参数:
    data -- 输入数据，形状为(N, D)
    window_size -- 窗口长度W
    stride -- 窗口移动步长S
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
        with torch.no_grad():
            processed_window = model(window_data).detach().cpu().numpy()
        # 将处理结果累加到最终结果
        result[start:end] += processed_window
        # 更新计数
        count[start:end] += 1
    # 计算平均值（重叠部分取平均）
    result /= count[:, np.newaxis]
    return result



def inference_pt(dataset_root, model_name, weight_path, visualization_save_root, min_cluster_size=10, slice_len=3000, stride=2500):
    model = get_model(model_name, weight_path).cuda().eval()

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    # clusterer2 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size2)
    for filename in tqdm(os.listdir(dataset_root)):
        data = torch.load(os.path.join(dataset_root, filename))

        freqs = normalize_zscore(data[:, 0])
        pws = normalize_zscore(data[:, 1])
        pas = normalize_zscore(data[:, 2])
        toas = data[:, 3]
        # dtoa = normalize_zscore(data[:, 4])
        dtoa = normalize_zscore(F.pad(torch.diff(toas), pad=(1, 0), value=0.))
        data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unsqueeze(0).float().cuda()
        label_gt = data[:, -1]

        output = segment_inference(data_input, slice_len, stride, model)
        output = F.normalize(output, p=2, dim=-1)
        cluster_labels = clusterer.fit_predict(output)

        # TODO: Case2 : generate cluster results of each slice, and get the cluster center, cluster all the centers

        print('{}: {} pred_targets,  {} true_targets'.format(filename, len(np.unique(cluster_labels)), len(torch.unique(label_gt))))
        data_input = data_input.detach().cpu().numpy()
        pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None, Tag_SampleRates=None)
        pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                 TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None,
                                 Tag_SampleRates=None)
        draw_pdwtrain_with_label(pdw_train_gt,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_gt.png')))
        draw_pdwtrain_with_label(pdw_train_out,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_pred.png')))

        # show T-SNE
        output_tSNE = t_sne(output)
        plt.figure(figsize=(12, 8))
        plt.title('Output T-SNE')
        plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=label_gt)
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')
        plt.tight_layout()
        plt.savefig(filename.replace('.pt', '.png'), dpi=200)
        # plt.show()
        plt.close()





def inference_h5(dataset_root, model_name, weight_path, visualization_save_root, min_cluster_size=10, slice_len=3000, stride=2500):
    model = get_model(model_name, weight_path).cuda().eval()

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    # clusterer2 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size2)
    for filename in tqdm(os.listdir(dataset_root)):
        # data = torch.load(os.path.join(dataset_root, filename))
        data = read_pdw_new_recog(os.path.join(dataset_root, filename))
        freqs = torch.from_numpy(normalize_zscore(np.array(data.Freqs)))
        pws = torch.from_numpy(normalize_zscore(np.array(data.PWs)))
        pas = torch.from_numpy(normalize_zscore(np.array(data.PAs)))
        dtoa = torch.from_numpy(normalize_zscore(np.array(data.update_dtoa())))
        # dtoa = normalize_zscore(data[:, 4])

        data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unsqueeze(0).float().cuda()
        label_gt = data[:, -1]

        output = segment_inference(data_input, slice_len, stride, model)
        output = F.normalize(output, p=2, dim=-1)
        cluster_labels = clusterer.fit_predict(output)

        # TODO: Case2 : generate cluster results of each slice, and get the cluster center, cluster all the centers

        print('{}: {} pred_targets,  {} true_targets'.format(filename, len(np.unique(cluster_labels)), len(torch.unique(label_gt))))
        data_input = data_input.detach().cpu().numpy()
        pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None, Tag_SampleRates=None)
        pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                 TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None,
                                 Tag_SampleRates=None)
        draw_pdwtrain_with_label(pdw_train_gt,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_gt.png')))
        draw_pdwtrain_with_label(pdw_train_out,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_pred.png')))

        # show T-SNE
        output_tSNE = t_sne(output)
        plt.figure(figsize=(12, 8))
        plt.title('Output T-SNE')
        plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=label_gt)
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')
        plt.tight_layout()
        plt.savefig(filename.replace('.pt', '.png'), dpi=200)
        # plt.show()
        plt.close()

if __name__ == '__main__':
    N = 6
    D = 2
    data = torch.rand(N, D)
    def func(x):
        noise = np.random.rand() * 10
        print('noise', noise)
        return x + noise
    result = segment_inference(data, window_size=5, stride=3, model=func)
    print(data)
    print(result)
    # np.random.seed(42)
    # inference_pt(dataset_root=r'F:\Datasets\JHT2025Pre\新建文件夹',
    #           model_name='Flowformer',
    #           weight_path=r'F:\github\JHT2025\experiments\023_Deinterleaving_Flowformer_CL_RoPE_2to10_1k_allNeg_LossImpl3_wsl\models\net_g_100000.pth',
    #           visualization_save_root=r'./2to10_Folwfomer',
    #           slice_len=5000)
