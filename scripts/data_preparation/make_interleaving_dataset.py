import numpy as np
import os
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

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


def select_files_by_specific_aaaa(
        aaaa_list: List[str],
        directory_path: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        pattern: str = r"S_(\w{4})_\w{5}_\w{5}",
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    从文件中选出指定AAAA列表中的文件，每个AAAA随机均匀选择一个文件

    参数:
    aaaa_list: 必须选择的AAAA代码列表
    directory_path: 目录路径，程序会读取该目录下所有匹配模式的文件
    file_list: 文件列表，如果不提供目录路径，则使用此文件列表
    pattern: 文件名模式，默认匹配 S_AAAA_BBBBB_CCCCC 格式
    ensure_even_distribution: 是否确保均匀分布（当文件数不同时）
    max_retries: 最大重试次数（用于确保均匀性）

    返回:
    (选中的文件列表, 每个AAAA对应的所有文件字典)
    """
    # 验证AAAA列表
    if len(aaaa_list) == 0:
        raise ValueError("AAAA列表不能为空")

    # 转换AAAA列表为集合，去重
    target_aaaa_set = set(str(code).strip() for code in aaaa_list)
    # print(f"目标AAAA代码: {sorted(target_aaaa_set)}")

    # 获取文件列表
    if directory_path:
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {directory_path}")

        # 使用正则表达式匹配文件名
        pattern_regex = re.compile(pattern)
        files = [str(f.name) for f in directory.iterdir()
                 if f.is_file() and pattern_regex.match(f.name)]
    elif file_list:
        # 使用提供的文件列表
        pattern_regex = re.compile(pattern)
        files = [f for f in file_list if pattern_regex.match(f)]
    else:
        raise ValueError("必须提供 directory_path 或 file_list 参数")

    if not files:
        # print("没有找到匹配模式的文件")
        return [], {}

    # print(f"找到 {len(files)} 个匹配的文件")

    # 使用字典按AAAA分组文件
    all_file_groups = defaultdict(list)

    for filename in files:
        # 使用正则表达式提取AAAA部分
        match = pattern_regex.match(filename)
        if match:
            # 提取分组1，即AAAA部分
            aaaa_code = match.group(1)
            all_file_groups[aaaa_code].append(filename)

    if not all_file_groups:
        # print("无法从文件名中提取AAAA代码")
        return [], {}

    # print(f"找到 {len(all_file_groups)} 个不同的AAAA代码")

    # 筛选出目标AAAA的文件
    target_file_groups = {}
    for aaaa_code in target_aaaa_set:
        if aaaa_code in all_file_groups:
            target_file_groups[aaaa_code] = all_file_groups[aaaa_code]
        else:
            print(f"警告: 未找到AAAA代码为 '{aaaa_code}' 的文件")

    if not target_file_groups:
        # print("没有找到任何目标AAAA代码的文件")
        return [], dict(target_file_groups)

    # print(f"\n找到 {len(target_file_groups)}/{len(target_aaaa_set)} 个目标AAAA代码的文件")

    # 统计每个AAAA的文件数量
    # for aaaa_code, file_list in target_file_groups.items():
        # print(f"  AAAA '{aaaa_code}': {len(file_list)} 个文件")


    selected_files = []
    for aaaa_code, file_list in target_file_groups.items():
        if file_list:
            selected_file = random.choice(file_list)
            selected_files.append(selected_file)
            # print(f"\nAAAA代码 '{aaaa_code}': 选择了文件 '{selected_file}'")

    # print(f"\n总共选择了 {len(selected_files)} 个文件")
    return selected_files, dict(target_file_groups)


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
        pdw_train.TOAdots += int(offset)

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
    # 防止toa出现负值
    toa_min = sorted_toa.min()
    if toa_min < 0:
        sorted_toa += abs(toa_min)
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



def random_interleaving2(emitter_num, missing_uprate, path, return_single=False):
    info = {'emitters': [],
            'missing_rates': [],
            'pdw_nums': []
            }
    filenames = os.listdir(path)
    emitters = ['1111', '1112', '1121', '1131',
                '1132', '1141', '1151', '1311',
                '1421', '1422', '1451', '2111',
                '2151', '2211', '2421']

    selected_emitter = np.random.choice(emitters, emitter_num, replace=False)
    # print(selected_emitter)
    selected, _ = select_files_by_specific_aaaa(aaaa_list=selected_emitter, file_list=filenames)
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
        pdw_train.TOAdots += int(offset)

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
    # 防止toa出现负值
    toa_min = sorted_toa.min()
    if toa_min < 0:
        sorted_toa += abs(toa_min)
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
        # 如果切片中只剩下一个脉冲，则不保留
        if len(torch.unique(tensor[start:end][:, 5])) > 1:
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
    root = r'F:\Datasets\2025金海豚初赛数据\分选\单一'
    save_path = r'F:\Datasets\JHT2025Pre\RandomMixed'
    save_figure_path = r'F:\Datasets\JHT2025Pre\MixedFigure'
    save_slice_path = r'F:\Datasets\JHT2025Pre\MixedAndClipped'
    repeat_num = [5, 5, 5, 5, 5, 3, 3, 2, 2]
    missing_uprate = 0.0
    clipping_stride = 1000
    clipping_overlap = 0.2
    save_figure = True

    interleaving = True
    clipping = True
    if interleaving:
        for i, emitter_num in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
            for _ in tqdm(range(repeat_num[i])):
                merged, pdw_trains, info = random_interleaving2(emitter_num=emitter_num, missing_uprate=missing_uprate,
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
                torch.save(input_data, os.path.join(save_path, '{}_{}.pt').format(emitters, total_num))          # (N, 6)
                if save_figure:
                    draw_pdwtrain(merged, os.path.join(save_figure_path, '{}_{}.png').format(emitters, total_num))
                    # for emitter, missing_rate, pdw_train in zip(info['emitters'], info['missing_rates'], pdw_trains):
                    #     draw_pdwtrain(pdw_train,
                    #                   os.path.join(save_figure_path, '{}_{:.3f}.png').format(emitter, missing_rate))
    if clipping:
        make_clipping(save_path, save_slice_path, stride=clipping_stride, overlap=clipping_overlap, n_thread=4)
