import os.path
import pandas as pd
import numpy as np
from src import scandir
from src.data.data_util import read_pdw

def convert_deinterleaving(mode):
    if mode == '单一':
        root = r'G:\datasets\2025金海豚初赛数据\分选\混合（示例）'
        save_root = r'G:\datasets\2025金海豚初赛数据\分选\混合示例_txt'
    elif mode == '混合':
        root = r'G:\datasets\2025金海豚初赛数据\分选\单一'
        save_root = r'G:\datasets\2025金海豚初赛数据\分选\单一_txt'
    single_data = scandir(root)


    for data in single_data:
        pdw = read_pdw(os.path.join(root, data))
        df = {'TOA(s)': [], 'Param1': [], 'Param2': [], 'Param3': [], 'Param4': [], 'SgnIdx': []}
        TOADots = pdw.TOAdots / 1e9   # ns -> s
        df['TOA(s)'] = TOADots
        DTOA = np.diff(TOADots)
        DTOA = np.concatenate(([0], DTOA), axis=0)
        df['Param1'] = pdw.Freqs + pdw.Tag_CenterFreqs
        df['Param2'] = pdw.PWdots / pdw.Tag_SampleRates   # us
        df['Param3'] = DTOA
        df['Param4'] = pdw.PAs
        df['SgnIdx'] = pdw.Labels
        df['SampleRate'] = pdw.Tag_SampleRates
        df['CenterFreq'] = pdw.Tag_CenterFreqs
        df['Freqs'] = pdw.Freqs
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(save_root, data.replace('.h5','.txt')), index=False, sep='\t')

def convert_recognition(mode):
    if mode == '型号':
        root = r'G:\datasets\2025金海豚初赛数据\识别\型号识别'
        save_root = r'G:\datasets\2025金海豚初赛数据\识别\型号识别_txt'
    elif mode == '个体':
        root = r'G:\datasets\2025金海豚初赛数据\识别\个体识别5x5'
        save_root = r'G:\datasets\2025金海豚初赛数据\识别\个体识别5x5_txt'
    single_data = scandir(root)
    for data in single_data:
        pdw = read_pdw(os.path.join(root, data))
        df = {'TOA(us)': [], 'Param1': [], 'Param2': [], 'Param3': [], 'Param4': []}
        # df2 = {'IntraPulse': []}
        # df3 = {'Begin': [], 'End': []}
        TOADots = pdw.TOAdots / 1e3   # ns -> us
        df['TOA(us)'] = TOADots
        DTOA = np.diff(TOADots)
        DTOA = np.concatenate(([0], DTOA), axis=0)
        df['Param1'] = pdw.Freqs + pdw.Tag_CenterFreqs
        df['Param2'] = pdw.PWdots / pdw.Tag_SampleRates   # us
        df['Param3'] = DTOA
        df['Param4'] = pdw.PAs
        df['SgnIdx'] = pdw.Labels
        df['SampleRate'] = pdw.Tag_SampleRates
        df['CenterFreq'] = pdw.Tag_CenterFreqs
        df['Freqs'] = pdw.Freqs
        # df['label'] = pdw.Labels

        # df2['IntraPulse'] = pdw.IntraPulse
        # df3['Begin'] = pdw.Indexs
        # df3['End'] = pdw.PWdots + pdw.Indexs

        df = pd.DataFrame(df)
        # df2 = pd.DataFrame(df2)
        # df3 = pd.DataFrame(df3)

        df.to_csv(os.path.join(save_root, data.replace('.h5', '.txt')), index=False, sep=' ')
        # df2.to_csv(os.path.join(save_root, data.replace('.h5', '_intrapulse.txt')), index=False, sep=' ')
        # df3.to_csv(os.path.join(save_root, data.replace('.h5', '_intrapulse_index.txt')), index=False, sep=' ')

convert_deinterleaving('单一')
convert_deinterleaving('混合')
convert_recognition('个体')
convert_recognition('型号')