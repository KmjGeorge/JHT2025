import pandas as pd
import torch

from src import calculate_nmi, calculate_ac, calculate_ari
import os
root = r'F:\github\JHT2025\experiments\001_Deinterleaving_Reformer_CL_2to10\visualization'


for folder in os.listdir(root):
    df_gt = None
    df_pred = None
    for filename in os.listdir(os.path.join(root, folder)):
        if ('iter50000' in filename) and ('.csv' in filename):
            if 'gt' in filename:
                df_gt = pd.read_csv(os.path.join(root, folder, filename))
            else:
                df_pred = pd.read_csv(os.path.join(root, folder, filename))
        if df_gt is not None and df_pred is not None:
            pred = torch.Tensor(list(df_pred['0']))
            gt = torch.Tensor(list(df_gt['0']))
            print('File:', folder, end='\t')
            print(torch.unique(gt), end='\t')
            print('NMI:', calculate_nmi(pred, gt), end='\t')
            print('AC:', calculate_ac(pred, gt), end='\t')
            print('ARI', calculate_ari(pred, gt), end='\n\n')


# df_gt = pd.read_csv(os.path.join(root, '1111+1422+2151_51304_1', '1111+1422+2151_51304_1_iter50000_gt.csv'))
# df_pred = pd.read_csv(os.path.join(root, '1111+1422+2151_51304_1', '1111+1422+2151_51304_1_iter50000.csv'))
# pred = torch.Tensor(list(df_pred['0']))
# print(torch.unique(pred))
# gt = torch.Tensor(list(df_gt['0']))
# print('NMI:', calculate_nmi(pred, gt).item(), end='\t')
# print('AC:', calculate_ac(pred, gt).item(), end='\t')
# print('ARI', calculate_ari(pred, gt).item(), end='\n\n')
