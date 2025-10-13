import os
import random

import numpy as np

from src.data.data_util import read_pdw


def random_interleaving(emitter_num, path):
    filenames = os.listdir(path)
    selected = np.random.choice(filenames, emitter_num, replace=False)
    pdw_trains = []
    for filename in selected:
        print(filename)
        pdw_train = read_pdw(os.path.join(path, filename))
        print(pdw_train.get_data_range())
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
    delta_max = min_length * 0.25

    # 对每个列表进行随机偏移
    for idx, lst in enumerate(pdw_trains):
        min_t, max_t, length = list_ranges[idx]
        if length == 0:  # 单点列表
            C_i = min_t
        else:
            C_i = (min_t + max_t) / 2.0

        # 在允许范围内随机生成新中点
        T_i = random.uniform(T - delta_max, T + delta_max)
        offset = T_i - C_i

        # 应用偏移
        for data in lst:
            data.time += offset

    # 合并所有列表并排序
    merged_pdw_train = []
    for lst in pdw_trains:
        merged_pdw_train.extend(lst)
    merged_pdw_train.sort(key=lambda x: x.toa)

    return merged_pdw_train

if __name__ == '__main__':
    merged = random_interleaving(emitter_num=3, path=r'F:\Datasets\2025金海豚初赛数据\分选\单一完整')

