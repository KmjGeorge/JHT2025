from torch.utils import data as data
from src.utils.registry import DATASET_REGISTRY
from src.data.data_util import read_pdw_from_interleaved
import numpy as np
import torch
import os

# @DATASET_REGISTRY.register()
class DeinterleavingDataset(data.Dataset):
    """deinterleaving dataset with (pdws, labels) pairs where pdws in (N, 4) and labels in (N, 1)

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot (str): Data root path.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(DeinterleavingDataset, self).__init__()
        self.opt = opt
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.folder = opt['dataroot']
        self.data = []
        for filename in os.listdir(self.folder):
            self.data.append(read_pdw_from_interleaved(os.path.join(self.folder, filename)))


    def __getitem__(self, index):
        data_slice = self.data[index]
        freqs = data_slice.Freqs
        pws = data_slice.PWs
        pas = data_slice.PAs
        toadots = data_slice.TOAdots
        pdws = torch.from_numpy(np.stack([freqs, pws, pas, toadots], axis=1))
        print(data_slice.Labels)
        labels = torch.from_numpy(data_slice.Labels)
        return pdws, labels

    def __len__(self):
        return len(self.data)

dataset = DeinterleavingDataset(opt={'dataroot': r'G:\datasets\2025金海豚初赛数据\分选\混合（示例）'})
for pdw, label in dataset:
    print(pdw.shape, label.shape)