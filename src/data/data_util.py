import os

import numpy as np
import torch
from os import path as osp

from sklearn.manifold import TSNE
from torch.nn import functional as F
import h5py
import pandas as pd
from src.utils import img2tensor, scandir
import matplotlib.pyplot as plt
import matplotlib

#
# color_map_selected = []
# for color in list(matplotlib.colors.CSS4_COLORS):
#     if ('gray' in color) or ('white' in color) or ('black' in color):
#         continue
#     else:
#         color_map_selected.append(color)
# np.random.shuffle(color_map_selected)

color_map_selected = [
    'red', 'blue', 'green', 'orange', 'purple',
    'cyan', 'magenta', 'lime', 'navy', 'gold',
    'teal', 'crimson', 'indigo', 'olive', 'hotpink',
    'darkorange', 'dodgerblue', 'forestgreen', 'darkviolet', 'sienna',
    'deepskyblue', 'lawngreen', 'darkorchid', 'tomato', 'royalblue',
    'yellowgreen', 'mediumvioletred', 'darkturquoise', 'orangered', 'mediumblue'
]


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x


class PDWTrain:
    def __init__(self, Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots, missing_rate=0.,
                 IntraPulse=None):

        if Tag_CenterFreqs is not None:
            self.Tag_CenterFreqs = Tag_CenterFreqs
        if Tag_SampleRates is not None:
            self.Tag_SampleRates = Tag_SampleRates

        self.Freqs = Freqs
        self.PAs = PAs
        self.Labels = Labels
        self.PWs = PWs
        self.TOAdots = TOAdots
        self.missing_rate = missing_rate
        if IntraPulse is not None:
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

    def __len__(self):
        return len(self.TOAdots)

    def update_dtoa(self):
        self.DTOA = np.concatenate([[0], np.diff(self.TOAdots)], axis=0)
        return self.DTOA


def read_pdw(path):
    label_map = {'1111': 0,
                 '1112': 1,
                 '1121': 2,
                 '1131': 3,
                 '1132': 4,
                 '1141': 5,
                 '1151': 6,
                 '1311': 7,
                 '1421': 8,
                 '1422': 9,
                 '1451': 10,
                 '2111': 11,
                 '2151': 12,
                 '2211': 13,
                 '2421': 14}
    with h5py.File(path, 'r') as f:
        def print_all_keys(name, obj):
            print("文件中的所有键：")
            if isinstance(obj, (h5py.Dataset, h5py.Group)):
                print(name)

        # f.visititems(print_all_keys)
        Tag_Nums = np.array(f['TAG']['NUM']).squeeze()
        Tag_CenterFreqs = np.repeat(f['TAG']['CenterFreq'][0][0], Tag_Nums)
        Tag_SampleRates = np.repeat(f['TAG']['SampleRate'][0][0], Tag_Nums)
        Freqs = np.array(f['InterPulse']['Freq']).squeeze() + Tag_CenterFreqs
        PAs = np.array(f['InterPulse']['PA']).squeeze()
        try:
            Labels = np.repeat(int(f['InterPulse']['LABEL'][0][0]), Tag_Nums)
            for j, label in enumerate(Labels):
                Labels[j] = label_map[str(label)]
        except:
            Labels = None
        # Indexs = np.array(f['InterPulse']['INDEX']).squeeze()

        PWdots = np.array(f['InterPulse']['PWdot']).squeeze()
        PWs = PWdots / Tag_SampleRates  # us
        TOAdots = np.array(f['InterPulse']['TOAdot']).squeeze() / 1e3  # ns -> us
        try:
            IntraPulse = np.array(f['IntraPulse']['DATA']).squeeze()
            pdwtrain = PDWTrain(Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots, IntraPulse)
        except:
            pdwtrain = PDWTrain(Freqs, PAs, Labels, Tag_CenterFreqs, Tag_SampleRates, PWs, TOAdots)
        return pdwtrain


def draw_pdwtrain(pdw, save_path):
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
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        plt.show()
        plt.close()


def draw_pdwtrain_with_label(pdw, save_path):
    plt.figure(figsize=(24, 12))
    plt.subplot(221)
    plt.title('Freq')
    x = [i + 1 for i in range(len(pdw.Freqs))]
    plt.scatter(x, pdw.Freqs, c=pdw.Labels, s=0.3)

    plt.subplot(222)
    plt.title('DTOA(us)')
    dtoa = np.concatenate(([0], np.diff(pdw.TOAdots)))
    plt.scatter(x, dtoa, c=pdw.Labels, s=0.3)

    plt.subplot(223)
    plt.title('PW')
    plt.scatter(x, pdw.PWs, c=pdw.Labels, s=0.3)

    plt.subplot(224)
    plt.title('PA')
    plt.scatter(x, pdw.PAs, c=pdw.Labels, s=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()


def normalize_minmax(tensor):
    min_, max_ = tensor.min(), tensor.max()
    return (tensor - min_) / (max_ - min_)


def normalize_zscore(tensor):
    min_, max_ = tensor.min(), tensor.max()
    tensor = (tensor - min_) / (max_ - min_)
    mean_, std_ = tensor.mean(), tensor.std()
    return (tensor - mean_) / std_


def pdw_write(label, label_gt, data, out_feature, save_img_path, save_config):
    dir_name = os.path.abspath(os.path.dirname(save_img_path))
    os.makedirs(dir_name, exist_ok=True)

    if save_config['save_pdwimg']:
        freqs, pws, pas, toas = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        pdwtrain = PDWTrain(freqs, pas, label, None, None, pws, toas)
        pdwtrain_gt = PDWTrain(freqs, pas, label_gt, None, None, pws, toas)
        # save pdw visualization
        draw_pdwtrain_with_label(pdwtrain, save_img_path)
        draw_pdwtrain_with_label(pdwtrain_gt, save_img_path.replace('.png', '_gt.png'))

    if save_config['save_featureTSNE']:

        # save T-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
        feature_tsne = tsne.fit_transform(out_feature)
        plt.figure(figsize=(10, 6))
        plt.title('Output T-SNE')
        for lbl in np.unique(label_gt):
            feature_of_label = feature_tsne[label_gt == lbl, :]
            plt.scatter(feature_of_label[:, 0], feature_of_label[:, 1], c=color_map_selected[int(lbl)], s=0.5, label=str(int(lbl)))
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')
        plt.legend(loc='lower right')
        plt.savefig(save_img_path.replace('.png', '_feature.png'), dpi=300)
        plt.close()

    if save_config['save_label']:
        # save label csv
        df1 = {'label': label}
        df2 = {'label': label_gt}
        df1, df2 = pd.DataFrame(df1), pd.DataFrame(df2)
        df1.to_csv(save_img_path.replace('.png', '.csv'), index=True)
        df2.to_csv(save_img_path.replace('.png', '_gt.csv'), index=True)
