import os
import random
import re
import numpy as np


def random_missing(pdw_train, missing_rate=0.2):
    length = len(pdw_train)
    selected = int(length * (1 - missing_rate))
    selected_indices = np.random.choice(np.arange(length), selected, replace=False)
    selected_indices.sort()
    selected_freq = pdw_train.Freqs[selected_indices]
    selected_toa = pdw_train.TOAdots[selected_indices]
    selected_pw = pdw_train.PWs[selected_indices]
    selected_pa = pdw_train.PAs[selected_indices]
    selected_label = pdw_train.Labels[selected_indices]
    selected_tag_centerfreqs = pdw_train.Tag_CenterFreqs[selected_indices]
    selected_tag_samplerates = pdw_train.Tag_SampleRates[selected_indices]
    return PDWTrain(selected_freq, selected_pa, selected_label, selected_tag_centerfreqs, selected_tag_samplerates,
                    selected_pw,
                    selected_toa)


def random_interleaving(emitter_num, missing_uprate, path, return_single=False):
    info = {'emitters': [],
            'missing_rates': [],
            'pdw_nums': []
            }
    filenames = os.listdir(path)
    selected = np.random.choice(filenames, emitter_num, replace=False)
    pdw_trains = []
    pattern = re.compile(r'S_(\d\d\d\d)_')
    for filename in selected:
        pdw_train = read_pdw(os.path.join(path, filename))
        missing_rate = np.random.uniform(0, missing_uprate)
        pdw_train = random_missing(pdw_train, missing_rate=missing_rate)
        signal_label = re.match(pattern, filename).group(1)

        info['emitters'].append(signal_label)
        info['missing_rates'].append(missing_rate)
        info['pdw_nums'].append(len(pdw_train))
        # print(filename, end='\t')
        # print('missing_rate =', missing_rate, end='\t')
        # print(pdw_train.get_data_range())
        pdw_trains.append(pdw_train)
    # 计算每个列表的时间范围和长度
    list_ranges = []
    for pdw_train in pdw_trains:
        max_toa, min_toa = pdw_train.get_data_range()['toa']
        length = max_toa - min_toa
        list_ranges.append((min_toa, max_toa, length))

    # 计算最小长度（包括0长度）
    min_length = min(length for _, _, length in list_ranges)

    # 设定公共中心点和最大偏移范围
    T = 0
    delta_max = min_length * 0.8

    # 对每个列表进行随机偏移
    for idx, pdw_train in enumerate(pdw_trains):
        min_t, max_t, length = list_ranges[idx]
        if length == 0:  # 单点列表
            C_i = min_t
        else:
            C_i = (min_t + max_t) / 2.0

        # 在允许范围内随机生成新中点
        T_i = random.uniform(T - delta_max, T + delta_max)
        offset = T_i - C_i

        # 应用偏移
        pdw_train.TOAdots += offset

    merged_freq, merged_toa, merged_pw, merged_pa, merged_label, merged_tag_centerfreqs, merged_tag_samplerates = np.array(
        []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for pdw_train in pdw_trains:
        merged_freq = np.concatenate([pdw_train.Freqs, merged_freq], axis=0)
        merged_toa = np.concatenate([pdw_train.TOAdots, merged_toa], axis=0)
        merged_pw = np.concatenate([pdw_train.PWs, merged_pw], axis=0)
        merged_pa = np.concatenate([pdw_train.PAs, merged_pa], axis=0)
        merged_label = np.concatenate([pdw_train.Labels, merged_label], axis=0)
        merged_tag_centerfreqs = np.concatenate([pdw_train.Tag_CenterFreqs, merged_tag_centerfreqs], axis=0)
        merged_tag_samplerates = np.concatenate([pdw_train.Tag_SampleRates, merged_tag_samplerates], axis=0)

    sorted_indices = sorted(range(len(merged_toa)), key=lambda i: merged_toa[i])
    sorted_freq = merged_freq[sorted_indices]
    sorted_toa = merged_toa[sorted_indices]
    sorted_pw = merged_pw[sorted_indices]
    sorted_pa = merged_pa[sorted_indices]
    sorted_label = merged_label[sorted_indices]
    sorted_tag_centerfreqs = merged_tag_centerfreqs[sorted_indices]
    sorted_tag_samplerates = merged_tag_samplerates[sorted_indices]
    if return_single:
        return PDWTrain(sorted_freq, sorted_pa, sorted_label, sorted_tag_centerfreqs, sorted_tag_samplerates, sorted_pw,
                        sorted_toa), pdw_trains, info

    else:
        return PDWTrain(sorted_freq, sorted_pa, sorted_label, sorted_tag_centerfreqs, sorted_tag_samplerates, sorted_pw,
                        sorted_toa), info


def make_clipping(path, save_path, stride=3000, overlap=0.2, n_thread=10):
    filenames = os.listdir(path)
    for filename in tqdm(filenames):
        data = torch.load(os.path.join(path, filename))
        slices = sliding_window_slice(data, stride, overlap)
        for idx, slice in enumerate(slices):
            torch.save(slice.float(), os.path.join(save_path, filename.replace('.pt', '_{}.pt').format(idx + 1)))


def sliding_window_slice(tensor, stride, overlap):

    if stride <= 0:
        raise ValueError("步长必须为正整数")
    if not (0 <= overlap < 1):
        raise ValueError("重叠率必须在[0, 1)范围内")

    length = tensor.shape[0]

    # 计算起始索引
    start_indices = []
    start = 0
    while start < length - stride:
        start_indices.append(start)
        start += stride
        # 如果最后一个窗口超出范围，调整起始位置确保覆盖末尾
        if start + stride > length:
            start = length - stride

    slices = []

    for start in start_indices:
        end = min(start + stride, length)
        slices.append(tensor[start:end])

    return slices


def z_score(data):
    data_max, data_min = data.max(), data.min()
    data = (data - data_min) / (data_max - data_min)
    data_mean, data_std = data.mean(), data.std()
    data = (data - data_mean) / data_std
    return data


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from src.data.data_util import PDWTrain, read_pdw, draw_pdwtrain

    # matplotlib.use('TkAgg')
    root = r'F:\Datasets\2025金海豚初赛数据\分选\单一完整'
    save_path = r'F:\Datasets\JHT2025Pre\RandomMixed'
    save_figure_path = r'F:\Datasets\JHT2025Pre\MixedFigure'
    save_slice_path = r'F:\Datasets\JHT2025Pre\MixedAndClipped'
    repeat_num = 10
    missing_uprate = 0.2
    clipping_stride = 3000
    clipping_overlap = 0.2
    save_figure = False

    interleaving = True
    clipping = True
    if interleaving:
        for emitter_num in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for _ in tqdm(range(repeat_num)):
                merged, pdw_trains, info = random_interleaving(emitter_num=emitter_num, missing_uprate=missing_uprate,
                                                               path=root, return_single=True)
                emitters = '+'.join(info['emitters'])
                total_num = sum(info['pdw_nums'])
                freqs = merged.Freqs
                pws = merged.PWs
                pas = merged.PAs
                toas = merged.TOAdots
                dtoa = merged.update_dtoa()
                labels = merged.Labels

                input_data = torch.from_numpy(np.stack([freqs, pws, pas, toas, dtoa, labels], axis=1)).float()
                torch.save(input_data, os.path.join(save_path, '{}_{}.pt').format(emitters, total_num))
                if save_figure:
                    draw_pdwtrain(merged, os.path.join(save_figure_path, '{}_{}.png').format(emitters, total_num))
                    for emitter, missing_rate, pdw_train in zip(info['emitters'], info['missing_rates'], pdw_trains):
                        draw_pdwtrain(pdw_train, os.path.join(save_figure_path, '{}_{:.3f}.png').format(emitter, missing_rate))
    if clipping:
        make_clipping(save_path, save_slice_path, stride=clipping_stride, overlap=clipping_overlap, n_thread=4)
