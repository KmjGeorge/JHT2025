from torch.utils import data as data
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
        data = torch.load(os.path.join(self.folder, filename))
        pdws = data[:, :-1].float()
        labels = data[:, -1]
        return {'pdws': pdws, 'labels': labels}

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    dataset = DeinterleavingDataset(opt={'dataroot': r'G:\datasets\2025金海豚初赛数据\分选\随机混合切片'})
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for data in dataloader:
        print(data['pdws'].shape)
        print(data['labels'].shape)

