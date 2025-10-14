import torch
from sklearn.manifold import TSNE

from scripts.data_preparation.make_interleaving_dataset import draw_pdw_with_label, PDWTrain
from src.archs.samplemlp_arch import SimpleMLP
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import hdbscan

from src.data.data_util import normalize_zscore, normalize_minmax


def t_sne(feature):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=250)
    feature_tsne = tsne.fit_transform(feature)
    return feature_tsne


def inference(dataset_root, weight_path, visualization_save_root):
    model = SimpleMLP(seq_len=3000,
                      c_in=5,
                      c_out=32,
                      d_model=256).cuda()
    model.load_state_dict(torch.load(weight_path)['params'])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    for filename in tqdm(os.listdir(dataset_root)):
        data = torch.load(os.path.join(dataset_root, filename))

        freqs = normalize_zscore(data[:, 0])
        pws = normalize_zscore(data[:, 1])
        pas = normalize_zscore(data[:, 2])
        toas = normalize_minmax(data[:, 3])
        dtoa = normalize_zscore(data[:, 4])
        data_input = torch.stack([freqs, pws, pas, toas, dtoa], dim=1).float().cuda()
        label_gt = data[:, -1]

        output = model(data_input).detach().cpu().numpy()
        # print(output.shape)

        data_input = data_input.detach().cpu().numpy()
        pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None, Tag_SampleRates=None)

        cluster_labels = clusterer.fit_predict(output)
        pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                 TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None,
                                 Tag_SampleRates=None)
        draw_pdw_with_label(pdw_train_gt,
                            save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_gt.png')))
        draw_pdw_with_label(pdw_train_out,
                            save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_pred.png')))
        '''# show T-SNE
        output_tSNE = t_sne(output)
        orignal_tSNE = t_sne(pdw_train)
        plt.figure(figsize=(8, 12))
        plt.subplot(211)
        plt.title('Output T-SNE')
        plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=label)
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')

        plt.subplot(212)
        plt.title('Original T-SNE')
        plt.scatter(orignal_tSNE[:, 0], orignal_tSNE[:, 1], c=label)
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')

        plt.tight_layout()
        plt.savefig(filename.replace('.pt', '.png'), dpi=200)
        # plt.show()
        plt.close()
        '''


if __name__ == '__main__':
    np.random.seed(42)
    inference(dataset_root=r'F:\Datasets\JHT2025Pre\RandomMixed',
              weight_path=r'F:\github\JHT2025\experiments\001_Deinterleaving_Reformer_CL_SimpleMLP\models\net_g_15000.pth',
              visualization_save_root=r'./2to5')
