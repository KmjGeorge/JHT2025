import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from src.data.data_util import read_pdw, scandir

root = r'G:\datasets\2025金海豚初赛数据\分选\单一'
# root = r'C:\Users\KomeijiGeorge\OneDrive\桌面\AVIC\jht初赛测试集\慧眼识珠0928预赛数据\识别'
filenames = list(scandir(root))
pdws = []
for filename in filenames:
    pdws.append(read_pdw(os.path.join(root, filename)))

for filename, pdw in tqdm(zip(filenames, pdws)):
    plt.figure(figsize=(24, 12))

    toas = pdw.TOAdots
    dtoa = np.concatenate(([0], np.diff(toas)), axis=0)
    x_ = [i+1 for i in range(len(toas))]
    plt.subplot(221)
    plt.scatter(x=x_, y=pdw.Freqs, s=0.01, color='blue')
    plt.title('Freq')
    plt.ylabel('Freq')
    plt.xlabel('num')

    plt.subplot(222)
    plt.scatter(x=x_, y=dtoa, s=0.01, color='red')
    plt.title('DTOA')
    plt.ylabel('DTOA')
    plt.xlabel('num')


    plt.subplot(223)
    plt.scatter(x=x_, y=pdw.PWdots / pdw.Tag_SampleRates, s=0.01, color='purple')
    plt.title('PWdot')
    plt.ylabel('PW(us)')
    plt.xlabel('num')

    plt.subplot(224)
    plt.scatter(x=x_, y=pdw.PAs, s=0.01 ,color='green')
    plt.title('PA')
    plt.ylabel('PA')
    plt.xlabel('num')

    plt.tight_layout()
    plt.savefig('pdw_{}.png'.format(filename.replace('.h5', '')), dpi=300)