import os
import shutil
import numpy as np
from tqdm import tqdm
root = r'F:\Datasets\JHT2025Pre\MixedAndClipped'
save_train_root = r'F:\Datasets\JHT2025Pre\train1016'
save_val_root = r'F:\Datasets\JHT2025Pre\val1016'

if not os.path.exists(save_train_root):
    os.mkdir(save_train_root)
if not os.path.exists(save_val_root):
    os.mkdir(save_val_root)
filenames_all = os.listdir(root)
filenum = len(filenames_all)
# val_num = int(filenum * 0.1)
val_num = 100
filenames_val = np.random.choice(filenames_all, val_num, replace=False)
for filename in tqdm(filenames_all):
    if filename in filenames_val:
        shutil.move(os.path.join(root, filename), os.path.join(save_val_root, filename))
    else:
        shutil.move(os.path.join(root, filename), os.path.join(save_train_root, filename))

