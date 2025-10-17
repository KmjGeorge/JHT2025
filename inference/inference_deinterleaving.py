import torch
from sklearn.manifold import TSNE
# from src.archs.samplemlp_arch import SimpleMLP
from src.archs.Flowformer_arch import Flowformer

import os
import numpy as np
from tqdm import tqdm
import hdbscan
import matplotlib.pyplot as plt
from src.archs.Reformer_arch import Reformer
from src.archs.samplemlp_arch import SimpleMLP
from src.data.data_util import normalize_zscore, normalize_minmax, draw_pdwtrain_with_label, PDWTrain
import torch.nn.functional as F

def t_sne(feature):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=250)
    feature_tsne = tsne.fit_transform(feature)
    return feature_tsne


def get_model(model_name, weight_path):
    if model_name == 'SimpleMLP':
        model = SimpleMLP(seq_len=3000,
                          c_in=5,
                          c_out=32,
                          d_model=256).cuda()
    elif model_name == 'Flowformer':
        model = Flowformer(seq_len=230000,
                           enc_in=5,
                           d_model=128,
                           c_out=32,
                           n_heads=8,
                           d_ff=512,
                           activation='relu',
                           dropout=0.1,
                           e_layers=8,
                           pe_mode='RoPE')
    elif model_name == 'Reformer':
        model = Reformer(
            seq_len=2000,
            enc_in=5,
            d_model=128,
            c_out=32,
            n_heads=8,
            d_ff=512,
            activation='relu',
            dropout=0.1,
            e_layers=8,
            bucket_size=8,
            n_hashes=8)

    model.load_state_dict(torch.load(weight_path)['params'])
    model = model.cuda()
    return model


def inference(dataset_root, model_name, weight_path, visualization_save_root):
    model = get_model(model_name, weight_path)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    for filename in tqdm(os.listdir(dataset_root)):
        data = torch.load(os.path.join(dataset_root, filename))

        freqs = normalize_zscore(data[:, 0])
        pws = normalize_zscore(data[:, 1])
        pas = normalize_zscore(data[:, 2])
        toas = normalize_minmax(data[:, 3])
        # dtoa = normalize_zscore(data[:, 4])
        dtoa = F.pad(torch.diff(toas), pad=(1, 0), value=0.)
        data_input = torch.stack([freqs, pws, pas, toas, dtoa], dim=1).unsqueeze(0).float().cuda()
        label_gt = data[:, -1]

        output = model(data_input).detach().cpu().numpy().squeeze()

        data_input = data_input.detach().cpu().numpy()
        pdw_train_gt = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                TOAdots=data_input[:, 3], Labels=label_gt, Tag_CenterFreqs=None, Tag_SampleRates=None)

        cluster_labels = clusterer.fit_predict(output)
        print('{}: {} pred_targets,  {} true_targets'.format(filename, len(np.unique(cluster_labels)), len(torch.unique(label_gt))))
        '''
        pdw_train_out = PDWTrain(Freqs=data_input[:, 0], PWs=data_input[:, 1], PAs=data_input[:, 2],
                                 TOAdots=data_input[:, 3], Labels=cluster_labels, Tag_CenterFreqs=None,
                                 Tag_SampleRates=None)
        draw_pdwtrain_with_label(pdw_train_gt,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_gt.png')))
        draw_pdwtrain_with_label(pdw_train_out,
                                 save_path=os.path.join(visualization_save_root, filename.replace('.pt', '_pred.png')))
        '''
        '''
        # show T-SNE
        output_tSNE = t_sne(output)
        plt.figure(figsize=(12, 8))
        plt.title('Output T-SNE')
        plt.scatter(output_tSNE[:, 0], output_tSNE[:, 1], c=label_gt)
        plt.xlabel('Demension 1')
        plt.ylabel('Demension 2')
        plt.tight_layout()
        plt.savefig(filename.replace('.pt', '.png'), dpi=200)
        # plt.show()
        plt.close()
        '''


if __name__ == '__main__':
    np.random.seed(42)
    inference(dataset_root=r'F:\Datasets\JHT2025Pre\新建文件夹',
              model_name='Flowformer',
              weight_path=r'F:\github\JHT2025\experiments\023_Deinterleaving_Flowformer_CL_RoPE_2to10_1k_allNeg_LossImpl3_wsl\models\net_g_100000.pth',
              visualization_save_root=r'./2to10_Folwfomer')
