from cProfile import label

import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# from src.archs.samplemlp_arch import SimpleMLP
from src.archs.Flowformer_arch import Flowformer, Flowformer_P

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
from src import calculate_ac, calculate_nmi, calculate_ari
from scripts.pt2xml import pt_to_xml_deinterleaving


def t_sne(feature):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=250)
    feature_tsne = tsne.fit_transform(feature)
    return feature_tsne


def get_model(model_name, weight_path=None, param_key='params_ema'):
    if model_name == 'SimpleMLP':
        model = SimpleMLP(seq_len=3000,
                          c_in=5,
                          c_out=32,
                          d_model=256).cuda()
    elif model_name == 'Flowformer':
        model = Flowformer(seq_len=5000,
                           enc_in=5,
                           d_model=128,
                           c_out=32,
                           n_heads=8,
                           d_ff=512,
                           activation='relu',
                           dropout=0.1,
                           e_layers=8,
                           pe_mode='RoPE')
    elif model_name == 'Flowformer_P':
        model = Flowformer_P(seq_len=50000,
                             enc_in=4,
                             d_model=256,
                             c_out=128,
                             n_heads=8,
                             d_ff=512,
                             activation='relu',
                             dropout=0.1,
                             e_layers=8,
                             label_num=354,
                             prototype_num=64,
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
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path)[param_key])
    return model



def segment_inference(data, window_size, stride, model, c_out):
    """
    将数据分段处理并融合重叠部分
    参数:
    data -- 输入数据，形状为(N, D)
    window_size -- 窗口长度W
    stride -- 窗口移动步长S
    返回:
    result -- 最终处理结果，形状为(N, D)
    """
    b, n, c = data.shape
    # 验证参数有效性
    if window_size <= 0:
        raise ValueError("窗口大小必须大于0")
    if stride <= 0:
        raise ValueError("步长必须大于0")

    # 初始化结果数组和计数数组
    result = torch.zeros(size=(b, n, c_out), device=data.device)
    count = torch.zeros(size=(n, 1), dtype=torch.int, device=data.device)

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
        window_data = data[:, start:end, :]
        # 应用处理函数
        with torch.no_grad():
            ##### for prototype model the result is (out_feature, prototype)
            processed_window = model(window_data)[0]
        # 将处理结果累加到最终结果
        result[:, start:end, :] += processed_window
        # 更新计数
        count[start:end] += 1
    # 计算平均值（重叠部分取平均）
    result /= count
    return result

def segment_inference_slices(data, window_size, stride, model):
    """
    将数据分段处理并融合重叠部分
    参数:
    data -- 输入数据，形状为(N, D)
    window_size -- 窗口长度W
    stride -- 窗口移动步长S
    返回:
    slices, indices - 多个切片的处理结果和切片下标
    """
    b, n, c = data.shape
    # 验证参数有效性
    if window_size <= 0:
        raise ValueError("窗口大小必须大于0")
    if stride <= 0:
        raise ValueError("步长必须大于0")

    slices = []
    indices = []

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
        window_data = data[:, start:end, :]
        indices.append([start, end])
        # 应用处理函数
        with torch.no_grad():
            ##### for prototype model the result is (out_feature, prototype)
            processed_window = model(window_data)[0]
            slices.append(processed_window)

    return slices, indices



def inference(dataset_root, model_name, c_out, weight_path=None, param_key='params',
              visualization_save_root=None, xml_save_root=None, min_cluster_size=10, slice_len=3000, stride=2000,
              validation=True, mode='h5'):
    if visualization_save_root is not None:
        if not os.path.exists(visualization_save_root):
            os.mkdir(visualization_save_root)

    if xml_save_root is not None:
        if not os.path.exists(xml_save_root):
            os.mkdir(xml_save_root)

    model = get_model(model_name, weight_path, param_key).cuda().eval()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    for filename in tqdm(os.listdir(dataset_root)):
        if mode == 'h5':
            data = read_pdw_new_recog(os.path.join(dataset_root, filename), with_label=False)
            freqs = torch.from_numpy(normalize_zscore(np.array(data.Freqs)))
            pws = torch.from_numpy(normalize_zscore(np.array(data.PWs)))
            pas = torch.from_numpy(normalize_zscore(np.array(data.PAs)))
            dtoa = torch.from_numpy(normalize_zscore(np.array(data.update_dtoa())))
            data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unqueeze(0).float().cuda()
        elif mode == 'pt':
            data = torch.load(os.path.join(dataset_root, filename))
            freqs = normalize_zscore(data[:, 0])
            pws = normalize_zscore(data[:, 1])
            pas = normalize_zscore(data[:, 2])
            toas = data[:, 3]
            dtoa = normalize_zscore(F.pad(torch.diff(toas), pad=(1,0), value=0.))
            data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unsqueeze(0).float().cuda()
        else:
            raise ValueError('mode must be h5 or pt')

        output = segment_inference(data_input, slice_len, stride, model, c_out)
        output = F.normalize(output, p=2, dim=-1).detach().cpu().numpy()[0]

        colors = np.concatenate([[j for _ in range(1000)] for j in range(50)])[:len(output)]

        plt.scatter(output[:, 0], output[:, 1], s=0.5, c=colors)
        plt.savefig(os.path.join(visualization_save_root, filename.replace('.{}'.format(mode), '_mix_feature_1_2.png')), dpi=300)
        plt.close()

        cluster_labels = clusterer.fit_predict(output)
        cluster_num = max(cluster_labels) + 1
        if np.where(cluster_labels < 0):
            cluster_labels[np.where(cluster_labels < 0)] = cluster_num

        data_input = data_input.detach().cpu().numpy()[0]
        if xml_save_root is not None:
            pt_to_xml_deinterleaving(cluster_labels, os.path.join(xml_save_root, filename.replace('.{}'.format(mode), '_07.xml')))

        print(filename+' :', end='\t')

        if visualization_save_root is not None:
            if validation:
                if mode == 'pt':
                    label_gt = data[:, -1]
                elif mode == 'h5':
                    label_gt = data.Labels
                else:
                    raise ValueError('mode must be h5 or pt')
                print('{} true_targets'.format(len(torch.unique(label_gt))), end='\t')
                if visualization_save_root is not None:
                    pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2], TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None, Tag_SampleRates=None)
                    draw_pdwtrain_with_label(pdw_train_gt, save_path=os.path.join(visualization_save_root, filename.replace('.{}'.format(mode), '_gt.png')))

                # calculate_metrics
                ac = calculate_ac(torch.from_numpy(cluster_labels), torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
                nmi = calculate_nmi(torch.from_numpy(cluster_labels),
                                  torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
                ari = calculate_ari(torch.from_numpy(cluster_labels),
                                  torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
            print('{} pred_targets'.format(len(np.unique(cluster_labels))))
            pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2], TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None, Tag_SampleRates=None)

            draw_pdwtrain_with_label(pdw_train_out, save_path=os.path.join(visualization_save_root, filename.replace('.{}'.format(mode), '_pred.png')))

            # show T-SNE
            output_tSNE = t_sne(output)
            plt.figure(figsize=(12, 8))

            if validation:
                print('====== AC: {}, NMI: {}, ARI: {} ======'.format(ac, nmi, ari))
                plt.title('Output T-SNE (GT Label)')
                plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=label_gt, s=0.8)
            else:
                plt.title('Output T-SNE (Pred Label')
                plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=cluster_labels, s=0.8)
            plt.xlabel('Demension 1')
            plt.ylabel('Demension 2')
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_save_root, filename.replace('.{}'.format(mode), '_tsne.png')), dpi=300)
            plt.close()


def inference_hcluster(dataset_root, model_name, weight_path=None, param_key='params',
              visualization_save_root=None, xml_save_root=None, min_cluster_size1=10, min_cluster_size2=5, slice_len=3000, stride=2000,
              validation=True, mode='h5'):
    if visualization_save_root is not None:
        if not os.path.exists(visualization_save_root):
            os.mkdir(visualization_save_root)

    if xml_save_root is not None:
        if not os.path.exists(xml_save_root):
            os.mkdir(xml_save_root)

    model = get_model(model_name, weight_path, param_key).cuda().eval()
    clusterer1 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size1)
    clusterer2 = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size2)

    for filename in os.listdir(dataset_root):
        if mode == 'h5':
            data = read_pdw_new_recog(os.path.join(dataset_root, filename), with_label=False)
            freqs = torch.from_numpy(normalize_zscore(np.array(data.Freqs)))
            pws = torch.from_numpy(normalize_zscore(np.array(data.PWs)))
            pas = torch.from_numpy(normalize_zscore(np.array(data.PAs)))
            dtoa = torch.from_numpy(normalize_zscore(np.array(data.update_dtoa())))
            data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unqueeze(0).float().cuda()
        elif mode == 'pt':
            data = torch.load(os.path.join(dataset_root, filename))
            freqs = normalize_zscore(data[:, 0])
            pws = normalize_zscore(data[:, 1])
            pas = normalize_zscore(data[:, 2])
            toas = data[:, 3]
            dtoa = normalize_zscore(F.pad(torch.diff(toas), pad=(1, 0), value=0.))
            data_input = torch.stack([freqs, pws, pas, dtoa], dim=1).unsqueeze(0).float().cuda()
        else:
            raise ValueError('mode must be h5 or pt')

        outputs, segment_indices = segment_inference_slices(data_input, slice_len, stride, model)
        all_centers = []
        center_to_segment_map = []
        for i, output in enumerate(outputs):
            output = F.normalize(output, p=2, dim=-1)
            output = output.detach().cpu().numpy()[0]
            segment_labels = clusterer1.fit_predict(output)
            segment_labels[segment_labels == -1] = max(segment_labels) + 1

            centers = []
            for lbl_in_seg in np.unique(segment_labels):
                centers.append(clusterer1.weighted_cluster_medoid(lbl_in_seg))
            for j, center in enumerate(centers):
                all_centers.append(center)
                center_to_segment_map.append({
                    'segment_idx': i,
                    'center_idx': j,
                    'original_indices': segment_indices[i],
                    'segment_labels': segment_labels
                })

            all_centers = np.array(all_centers)

            if len(all_centers) > 1:
                scaler_centers = StandardScaler()
                centers_scaled = scaler_centers.fit_transform(all_centers)
                center_labels = clusterer2.fit_predict(centers_scaled)
            else:
                center_labels = np.array([0])

            final_labels = np.full(len(data_input[0]), -1, dtype=np.int32)
            current_label = 0
            label_mapping = {}
            for center_label in np.unique(center_labels):
                if center_label == -1:
                    continue
                label_mapping[center_label] = current_label
                current_label += 1

            for center_idx, center_info in enumerate(center_to_segment_map):
                seg_idx = center_info['segment_idx']
                start, end = center_info['original_indices']
                segment_labels = center_info['segment_labels']

                center_label = center_labels[center_idx]
                if center_label == -1:
                    final_label = -1
                else:
                    final_label = label_mapping[center_label]

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
                    segment_mask = (segment_labels == corresponding_cluster_label)
                    global_indices = np.arange(start, end)[segment_mask]
                    final_labels[global_indices] = final_label

            final_labels[final_labels == -1] = max(final_labels) + 1
            cluster_labels = final_labels
            # colors = np.concatenate([[j for _ in range(1000)] for j in range(50)])[:len(output)]

            data_input = data_input.detach().cpu().numpy()[0]
            if xml_save_root is not None:
                pt_to_xml_deinterleaving(cluster_labels,
                                         os.path.join(xml_save_root, filename.replace('.{}'.format(mode), '_07.xml')))

            print(filename + ' :', end='\t')
            if visualization_save_root is not None:
                if validation:
                    if mode == 'pt':
                        label_gt = data[:, -1]
                    elif mode == 'h5':
                        label_gt = data.Labels
                    else:
                        raise ValueError('mode must be h5 or pt')
                    print('{} true_targets'.format(len(torch.unique(label_gt))), end='\t')
                    if visualization_save_root is not None:
                        pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                                TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None,
                                                Tag_SampleRates=None)
                        draw_pdwtrain_with_label(pdw_train_gt, save_path=os.path.join(visualization_save_root,
                                                                                      filename.replace('.{}'.format(mode),
                                                                                                       '_gt.png')))

                    # calculate_metrics
                    ac = calculate_ac(torch.from_numpy(cluster_labels),
                                      torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
                    nmi = calculate_nmi(torch.from_numpy(cluster_labels),
                                        torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
                    ari = calculate_ari(torch.from_numpy(cluster_labels),
                                        torch.from_numpy(label_gt) if isinstance(label_gt, np.ndarray) else label_gt).item()
                print('{} pred_targets'.format(len(np.unique(cluster_labels))))
                pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                         TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None,
                                         Tag_SampleRates=None)

                draw_pdwtrain_with_label(pdw_train_out, save_path=os.path.join(visualization_save_root,
                                                                               filename.replace('.{}'.format(mode),
                                                                                                '_pred.png')))

            if validation:
                print('====== AC: {}, NMI: {}, ARI: {} ======'.format(ac, nmi, ari))



if __name__ == '__main__':
    np.random.seed(42)
    inference_hcluster(mode='h5',
                       dataset_root=r'',
                       model_name='Flowformer_P',
                       min_cluster_size1=40,
                       min_cluster_size2=3,
                       slice_len=2000,
                       stride=2000,
                       weight_path=r'',
                       param_key='params_ema',
                       visualization_save_root=r'./fs37_Flowformer_P_w2ks2k_hc',
                       xml_save_root=r'./fs37_Flowformer_P_w2ks2k_hc',
                       validation=False)
