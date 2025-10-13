import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt


class PDW:
    def __init__(self, Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots, IntraPulse=None):
        self.Freqs = Freqs
        self.PAs = PAs
        self.Labels = Labels
        # self.Indexs = Indexs-
        self.Tag_CenterFreqs = Tag_CenterFreqs
        self.Tag_SampleRates = Tag_SampleRates
        self.PWs = PWs
        self.TOAdots = TOAdots  # ns - > us
        self.IntraPulse = IntraPulse

    #  获取数据范围
    def get_data_range(self):
        result = {'freq': [self.Freqs.max(), self.Freqs.min()],
                  'pa': [self.PAs.max(), self.PAs.min()],
                  'pw': [self.PWs.max(), self.PWs.min()],
                  'toa': [self.TOAdots.max(), self.TOAdots.min()],
                  'center_freq': self.Tag_CenterFreqs,
                  }
        return result

    def draw_scatter(self, x_name, y_name, save_path, range=None):
        """
        Args:
            x_name: the x column name.
            y_name: the y column name.
            save_path: figure save path
            range: a tuple of data range, eg. (300, 800)
        Returns:
        """
        axis_name = {'Freq': self.Freqs,
                     'PA': self.PAs,
                     'PW': self.PWdots,
                     'TOA': self.TOAdots,
                     'freq': self.Freqs,
                     'pa': self.PAs,
                     'pw': self.PWdots,
                     'toa': self.TOAdots
                     }
        if range:
            x = axis_name[x_name][range[0]:range[1]]
            y = axis_name[y_name][range[0]:range[1]]
        else:
            x = axis_name[x_name]
            y = axis_name[y_name]
        # plt.xlim(axis_name[x_name].min(), axis_name[x_name].max())
        plt.ylim(axis_name[y_name].min(), axis_name[y_name].max())
        plt.scatter(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')


class PDW_Simplify:
    def __init__(self, Freqs, PAs, Labels, Indexs, Tag_CenterFreqs, Tag_Nums, Tag_SampleRates, PWdots, TOAdots,
                 IntraPulse=None):
        self.Freqs = Freqs
        self.PAs = PAs
        self.Labels = Labels
        self.Indexs = Indexs
        self.Tag_CenterFreqs = Tag_CenterFreqs
        self.Tag_Nums = Tag_Nums
        self.Tag_SampleRates = Tag_SampleRates
        self.PWdots = PWdots
        self.PWs = PWdots / Tag_SampleRates  # dot -> us
        self.TOAdots = TOAdots / 1e3  # ns - > us
        self.IntraPulse = IntraPulse

    #  获取数据范围
    def get_data_range(self):
        result = {'freq': [self.Freqs.max(), self.Freqs.min()],
                  'pa': [self.PAs.max(), self.PAs.min()],
                  'pw': [self.PWs.max(), self.PWs.min()],
                  'toa': [self.TOAdots.max(), self.TOAdots.min()],
                  'center_freq': self.Tag_CenterFreqs,
                  }
        return result

    def draw_scatter(self, x_name, y_name, save_path, range=None):
        """
        Args:
            x_name: the x column name.
            y_name: the y column name.
            save_path: figure save path
            range: a tuple of data range, eg. (300, 800)
        Returns:
        """
        axis_name = {'Freq': self.Freqs,
                     'PA': self.PAs,
                     'PW': self.PWdots,
                     'TOA': self.TOAdots,
                     'freq': self.Freqs,
                     'pa': self.PAs,
                     'pw': self.PWdots,
                     'toa': self.TOAdots
                     }
        if range:
            x = axis_name[x_name][range[0]:range[1]]
            y = axis_name[y_name][range[0]:range[1]]
        else:
            x = axis_name[x_name]
            y = axis_name[y_name]
        # plt.xlim(axis_name[x_name].min(), axis_name[x_name].max())
        plt.ylim(axis_name[y_name].min(), axis_name[y_name].max())
        plt.scatter(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')


def read_pdw(path):
    with h5py.File(path, 'r') as f:
        def print_all_keys(name, obj):
            print("文件中的所有键：")
            if isinstance(obj, (h5py.Dataset, h5py.Group)):
                print(name)

        # f.visititems(print_all_keys)
        Tag_Nums = np.array(f['TAG']['NUM']).squeeze()
        Freqs = np.array(f['InterPulse']['Freq']).squeeze()
        PAs = np.array(f['InterPulse']['PA']).squeeze()
        try:
            Labels = np.repeat(f['InterPulse']['LABEL'][0][0], Tag_Nums)
        except:
            Labels = None
        # Indexs = np.array(f['InterPulse']['INDEX']).squeeze()
        Tag_CenterFreqs = np.repeat(f['TAG']['CenterFreq'][0][0], Tag_Nums)
        Tag_SampleRates = np.repeat(f['TAG']['SampleRate'][0][0], Tag_Nums)
        PWdots = np.array(f['InterPulse']['PWdot']).squeeze()
        PWs = PWdots / Tag_SampleRates  # us
        TOAdots = np.array(f['InterPulse']['TOAdot']).squeeze() / 1e3  # ns -> us
        try:
            IntraPulse = np.array(f['IntraPulse']['DATA']).squeeze()
            pdw = PDW(Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots, IntraPulse)
        except:
            pdw = PDW(Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots)
        return pdw


def draw_pdw(pdw, save_path):
    plt.figure(figsize=(24, 12))
    plt.subplot(221)
    plt.title('Freq')
    x = [i + 1 for i in range(len(pdw.Freqs))]
    plt.scatter(x, pdw.Freqs, color='r', s=0.1)

    plt.subplot(222)
    plt.title('DTOA(us)')
    dtoa = np.concatenate(([0], np.diff(pdw.TOAdots)))
    plt.scatter(x, dtoa, color='purple', s=0.1)

    plt.subplot(223)
    plt.title('PW')
    plt.scatter(x, pdw.PWs, color='blue', s=0.1)

    plt.subplot(224)
    plt.title('PA')
    plt.scatter(x, pdw.PAs, color='g', s=0.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def random_interleaving(emitter_num, path):
    filenames = os.listdir(path)
    selected = np.random.choice(filenames, emitter_num, replace=False)
    pdw_trains = []
    for filename in selected:
        print(filename, end='\t')
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
    sorted_freq = [merged_freq[i] for i in sorted_indices]
    sorted_toa = [merged_toa[i] for i in sorted_indices]
    sorted_pw = [merged_pw[i] for i in sorted_indices]
    sorted_pa = [merged_pa[i] for i in sorted_indices]
    sorted_label = [merged_label[i] for i in sorted_indices]
    sorted_tag_centerfreqs = [merged_tag_centerfreqs[i] for i in sorted_indices]
    sorted_tag_samplerates = [merged_tag_samplerates[i] for i in sorted_indices]
    return PDW(sorted_freq, sorted_pa, sorted_label, sorted_tag_centerfreqs, sorted_tag_samplerates, sorted_pw,
               sorted_toa), pdw_trains


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    merged, (pdw1, pdw2, pdw3, pdw4, pdw5, pdw6) = random_interleaving(emitter_num=6, path=r'F:\Datasets\2025金海豚初赛数据\分选\单一完整')
    draw_pdw(merged, 'merged.png')
    draw_pdw(pdw1, 'pdw1.png')
    draw_pdw(pdw2, 'pdw2.png')
    draw_pdw(pdw3, 'pdw3.png')
    draw_pdw(pdw4, 'pdw4.png')
    draw_pdw(pdw5, 'pdw5.png')
    draw_pdw(pdw6, 'pdw6.png')
