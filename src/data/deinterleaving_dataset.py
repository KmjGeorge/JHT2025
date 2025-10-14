from torch.utils import data as data

from src.data.data_util import normalize_zscore, normalize_minmax
from src.utils.registry import DATASET_REGISTRY
import torch
import os
from torch.utils.data import DataLoader


@DATASET_REGISTRY.register()
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
        self.filenames = os.listdir(self.folder)

    def __getitem__(self, index):
        filename = self.filenames[index]
        data_path = os.path.join(self.folder, filename)
        data = torch.load(data_path)

        # normalization internally each pdw train
        # for toa rescale to 0~1; for pw, pa, freq apply z-score
        pdws_nonorm = data[:, :-1]
        freqs = normalize_zscore(data[:, 0])
        pws = normalize_zscore(data[:, 1])
        pas = normalize_zscore(data[:, 2])
        toas = normalize_minmax(data[:, 3])
        dtoa = normalize_zscore(data[:, 4])
        # print(freqs.shape, pws.shape, pas.shape, toas.shape, dtoa.shape)
        pdws = torch.stack([freqs, pws, pas, toas, dtoa], dim=1).float()  # (N, 5)   freq, pw ,pa, toa, dtoa
        labels = data[:, 5]
        return {'pdws': pdws, 'pdws_nonorm': pdws_nonorm, 'labels': labels, 'data_path': data_path}

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    dataset = DeinterleavingDataset(opt={'dataroot': r'G:\datasets\2025金海豚初赛数据\分选\随机混合切片'})
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for data in dataloader:
        print(data['pdws'].shape)
        print(data['labels'].shape)
