import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F
import h5py
from src.data.transforms import mod_crop
from src.utils import img2tensor, scandir
import matplotlib.pyplot as plt

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

class PDW:
    def __init__(self, Freqs, PAs, Labels, Indexs, Tag_CenterFreqs, Tag_Nums, Tag_SampleRates, PWdots, TOAdots, IntraPulse=None):
        self.Freqs = Freqs
        self.PAs = PAs
        self.Labels = Labels
        self.Indexs = Indexs
        self.Tag_CenterFreqs = Tag_CenterFreqs
        self.Tag_Nums = Tag_Nums
        self.Tag_SampleRates = Tag_SampleRates
        self.PWdots = PWdots
        self.PWs = PWdots / Tag_SampleRates  # dot -> us
        self.TOAdots = TOAdots / 1e3   # ns - > us
        self.IntraPulse = IntraPulse

    def __len__(self):
        return len(self.TOAdots)
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

        Freqs = np.array(f['InterPulse']['Freq']).squeeze()
        PAs = np.array(f['InterPulse']['PA']).squeeze()
        try:
            Labels = np.array(f['InterPulse']['LABEL']).squeeze()
        except:
            Labels = None
        Indexs = np.array(f['InterPulse']['INDEX']).squeeze()
        Tag_CenterFreqs = np.array(f['TAG']['CenterFreq']).squeeze()
        Tag_Nums = np.array(f['TAG']['NUM']).squeeze()
        Tag_SampleRates = np.array(f['TAG']['SampleRate']).squeeze()
        PWdots = np.array(f['InterPulse']['PWdot']).squeeze()
        TOAdots = np.array(f['InterPulse']['TOAdot']).squeeze()
        try:
            IntraPulse = np.array(f['IntraPulse']['DATA']).squeeze()
            pdw = PDW(Freqs, PAs, Labels, Indexs, Tag_CenterFreqs, Tag_Nums, Tag_SampleRates, PWdots, TOAdots, IntraPulse)
        except:
            pdw = PDW(Freqs, PAs, Labels, Indexs, Tag_CenterFreqs, Tag_Nums, Tag_SampleRates, PWdots, TOAdots)
        return pdw


def read_pdw_from_interleaved(path):
    with h5py.File(path, 'r') as f:
        Freqs = np.array(f['InterPulse']['Freq']).squeeze()
        PAs = np.array(f['InterPulse']['PA']).squeeze()
        Labels = np.array(f['InterPulse']['LABEL']).squeeze()
        PWs = np.array(f['InterPulse']['PWdot']).squeeze()     ### us ###
        TOAdots = np.array(f['InterPulse']['TOAdot']).squeeze()
        pdw = PDW_Interleaved(Freqs, PAs, Labels, PWs, TOAdots)
        return pdw


class PDW_Interleaved:
    def __init__(self, Freqs, PAs, Labels, PWs, TOAdots):
        self.Freqs = Freqs
        self.PAs = PAs
        self.Labels = Labels
        self.PWs = PWs
        self.TOAdots = TOAdots

    def update_dtoa(self):
        self.DTOAs = np.concatenate([[0], np.diff(self.TOAdots)])
