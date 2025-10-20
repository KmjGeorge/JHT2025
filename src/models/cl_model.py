import hdbscan
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
from src.archs import build_network
from src.losses import build_loss
from src.metrics import calculate_metric
from src.utils import get_root_logger
from src.utils.registry import MODEL_REGISTRY
from src.models.base_model import BaseModel
from src.data.data_util import pdw_write
import torch.nn.functional as F

@MODEL_REGISTRY.register()
class CLModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(CLModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        try:
            self.net_h = build_network(self.opt['network_h'])  # 分类头，直接预测标签并计算CELoss
            self.net_h = self.model_to_device(self.net_h)
            self.print_network(self.net_h)
        except:
            self.net_h = None

        try:
            self.net_d = build_network(self.opt['network_d'])  # 重建头，重构输入并计算L1
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
        except:
            self.net_d = None

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_h = self.opt['path'].get('pretrain_network_h', None)
        if load_path_h is not None:
            param_key = self.opt['path'].get('param_key_h', 'params')
            self.load_network(self.net_h, load_path_h, self.opt['path'].get('strict_load_h', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        if self.net_h is not None:
            self.net_h.train()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('infonce_opt'):
            self.cri_infonce = build_loss(train_opt['infonce_opt']).to(self.device)
        else:
            self.cri_infonce = None

        if train_opt.get('seqce_opt'):
            self.cri_seqce = build_loss(train_opt['seqce_opt']).to(self.device)
        else:
            self.cri_seqce = None

        if train_opt.get('recon_opt'):
            self.cri_recon = build_loss(train_opt['recon_opt']).to(self.device)
        else:
            self.cri_recon = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.input = data['pdws'].to(self.device)  # (B, N, 5) x  noTOA  (B, N, 4)
        self.label = data['labels'].to(self.device)  # (B, N)
        self.input_nonorm = data['pdws_nonorm']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.input)  # B, L, D

        l_total = 0
        loss_dict = OrderedDict()

        if len(torch.unique(self.label)) == 1:  # 仅为单一脉冲时使用重构损失
            if self.cri_recon:
                recon = self.net_d(self.output)  # 不重构TOA，只重构DTOA
                l_recon = self.cri_recon(recon, self.input[:, (1, 2, 4)])
                l_total += l_recon
                loss_dict['l_Recon'] = l_recon
        else:
            if self.cri_recon:
                l_recon = 0
                loss_dict['l_Recon'] = l_recon

            if self.cri_infonce:
                l_cl = self._calculate_contrastive_loss_point_segment()
                l_total += l_cl
                loss_dict['l_InfoNCE'] = l_cl

            if self.cri_seqce:
                pred_logits = self.net_h(self.output)
                l_seqce = self.cri_seqce(pred_logits, self.label)
                l_total += l_seqce
                loss_dict['l_SeqCE'] = l_seqce

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.input)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.input)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='pulse')

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)

        for idx, val_data in enumerate(dataloader):
            data_name = osp.splitext(osp.basename(val_data['data_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            out_fea = self.output.squeeze(0).detach().cpu().numpy()
            cluster_labels = clusterer.fit_predict(out_fea)
            cluster_num = max(cluster_labels) + 1
            # 若存在标签为-1的离群点，将其视为一个新类
            if np.where(cluster_labels < 0):
                cluster_labels[np.where(cluster_labels < 0)] = cluster_num + 1

            metric_data['pred_labels'] = torch.from_numpy(cluster_labels)
            metric_data['true_labels'] = self.label.squeeze(0).detach().cpu()  # (1, B, N)
            if save_img['enable']:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], data_name,
                                             f'{data_name}_iter{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{data_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{data_name}_{self.opt["name"]}.png')
                pdw_write(metric_data['pred_labels'].numpy(), metric_data['true_labels'].numpy(),
                          self.input_nonorm.squeeze(0).detach().cpu().numpy(), out_fea, save_img_path, save_img)

            # tentative for out of GPU memory
            del self.output
            del self.label
            del self.input
            del self.input_nonorm
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {data_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_h is not None:
            self.save_network(self.net_h, 'net_h', current_iter)
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def _calculate_contrastive_loss(self):
        B, N, D = self.output.shape
        l_cl_batch_avg = 0
        for output, label in zip(self.output, self.label):  # for batch
            label_unique, counts = torch.unique(label, return_counts=True)  # 所有label种类
            label_cnt = 0
            l_cl_label_avg = 0

            for label_elem, count in zip(label_unique, counts):  # 对每种label，随机挑选出一个样本作为锚点，另一个作为正样本，并把其他label的特征作为负样本
                # 过滤样本数少于2的标签的样本
                if count < 2:
                    continue
                label_cnt += 1

                mask = (label == label_elem)
                feature = output[label == label_elem]  # (N1, D)  N1为该类的脉冲总数

                shuffle_idx = torch.randperm(feature.shape[0])  # 将该类的样本随机平均切分，一半为锚点，另一半为正样本
                mid = len(shuffle_idx) // 2
                anchor_idx, positive_idx = shuffle_idx[:mid], shuffle_idx[mid:2 * mid]
                anchor = feature[anchor_idx, :]  # (N1 // 2, D)
                positive = feature[positive_idx, :]  # (N1 // 2, D)

                negative = output[~mask, :]  # (N2, D)  N2为其他类的脉冲总数    此时InfoNCE应为unpaired模式

                l_cl = self.cri_infonce(query=anchor, positive_key=positive, negative_keys=negative)
                l_cl_label_avg += l_cl
            l_cl_label_avg = l_cl_label_avg / label_cnt
            l_cl_batch_avg += l_cl_label_avg
        l_cl_batch_avg /= B

        return l_cl_batch_avg

    # def _calculate_contrastive_loss_2(self):
    #     '''
    #     展平为 (B*N, D) 加快速度
    #     '''
    #     B, N, D = self.output.shape
    #     output = self.output.reshape(B * N, D)
    #     label = self.label.reshape(B * N)
    #
    #     label_unique, counts = torch.unique(label, return_counts=True)  # 所有label种类
    #     label_cnt = 0
    #     l_cl_label_avg = 0
    #     for label_elem, count in zip(label_unique, counts):  # 对每种label，随机挑选出一个样本作为锚点，另一个作为正样本，并把其他label的特征作为负样本
    #         # 过滤样本数少于2的标签的样本
    #         if count < 2:
    #             continue
    #         label_cnt += 1
    #
    #         mask = (label == label_elem)
    #         feature = output[label == label_elem]  # (N1, D)  N1为该类的脉冲总数
    #
    #         shuffle_idx = torch.randperm(feature.shape[0])  # 将该类的样本随机平均切分，一半为锚点，另一半为正样本
    #         mid = len(shuffle_idx) // 2
    #         anchor_idx, positive_idx = shuffle_idx[:mid], shuffle_idx[mid:2 * mid]
    #         anchor = feature[anchor_idx, :]  # (N1 // 2, D)
    #         positive = feature[positive_idx, :]  # (N1 // 2, D)
    #
    #         negative = output[~mask, :]  # (N2, D)  N2为其他类的脉冲总数    此时InfoNCE应为unpaired模式
    #
    #         l_cl = self.cri_infonce(query=anchor, positive_key=positive, negative_keys=negative)
    #         l_cl_label_avg += l_cl
    #     l_cl_label_avg /= label_cnt
    #
    #     return l_cl_label_avg

    # def _calculate_contrastive_loss_segment(self):
    #     """
    #     对于每个类别，构建一个全局平均特征，对比学习在点与平均特征上进行
    #     """
    #     B, N, D = self.output.shape
    #     output = self.output.reshape(B * N, D)
    #     label = self.label.reshape(B * N)
    #
    #     label_unique, counts = torch.unique(label, return_counts=True)  # 所有label种类
    #     label_cnt = 0
    #     l_cl_label_avg = 0
    #
    #     features_of_label = {}
    #     features_of_label_mean = {}
    #     for label_elem, count in zip(label_unique, counts):  # 对每种label，随机挑选出一个样本作为锚点，平均特征作为正样本，并把其他label的平均特征作为负样本
    #         # 过滤样本数少于2的标签的样本
    #         if count < 2:
    #             continue
    #         label_cnt += 1
    #         feature = output[label == label_elem]  # (N1, D)  N1为该类的脉冲总数
    #         features_of_label[label_elem.item()] = feature
    #         features_of_label_mean[label_elem.item()] = feature.mean(dim=0)
    #
    #     for label_elem, count in zip(label_unique, counts):
    #         if count < 2:
    #             continue
    #         feature = features_of_label[label_elem.item()]
    #         anchor_idx = torch.randperm(feature.shape[0])[0]  # 随机取出一个为锚点
    #         anchor = feature[anchor_idx, :]  # (1, D)
    #         positive = features_of_label_mean[label_elem.item()]
    #
    #         negative = torch.stack([value for key, value in features_of_label_mean.items() if key != label_elem.item()], dim=0)
    #
    #         l_cl = self.cri_infonce(query=anchor, positive_key=positive, negative_keys=negative)
    #         l_cl_label_avg += l_cl
    #     l_cl_label_avg /= label_cnt
    #
    #     return l_cl_label_avg

    def _calculate_contrastive_loss_point_segment(self):
        """
        联合使用点级和全局级损失对比损失
        """

        B, N, D = self.output.shape
        l_cl_batch_avg = 0
        for output, label in zip(self.output, self.label):  # for batch
            label_unique, counts = torch.unique(label, return_counts=True)  # 所有label种类
            label_cnt = 0
            l_cl_label_avg = 0

            features_of_label = {}  # 存储每个label的特征
            features_of_label_mean = {}  # 存储每个label的平均特征
            for label_elem, count in zip(label_unique, counts):
                # 过滤样本数少于2的标签的样本
                if count < 2:
                    continue
                label_cnt += 1
                feature = output[label == label_elem]  # (N1, D)  N1为该类的脉冲总数
                features_of_label[label_elem.item()] = feature
                features_of_label_mean[label_elem.item()] = feature.mean(dim=0, keepdim=True)

            for label_elem, count in zip(label_unique, counts):
                if count < 2:
                    continue
                mask = (label == label_elem)
                feature = features_of_label[label_elem.item()]  # 取出该label的特征
                shuffle_idx = torch.randperm(feature.shape[0])

                # 点级
                mid = len(shuffle_idx) // 2
                anchor_point_idx, positive_point_idx = shuffle_idx[:mid], shuffle_idx[mid:2 * mid]  # 一半为锚点，一半为正样本
                anchor_point = feature[anchor_point_idx, :]
                positive_point = feature[positive_point_idx, :]
                negative_point = output[~mask, :]  # 所有负样本

                # 片段级
                anchor_global = feature[shuffle_idx[0], :].unsqueeze(0)  # 随机取出一个为锚点  (1, D)
                positive_global = features_of_label_mean[label_elem.item()]  # 该类的平均特征为正样本
                negative_global = torch.cat(
                    [value for key, value in features_of_label_mean.items() if key != label_elem],
                    dim=0)  # 其他类的平均特征为负样本

                l_cl_p = self.cri_infonce(query=anchor_point, positive_key=positive_point, negative_keys=negative_point)
                l_cl_g = self.cri_infonce(query=anchor_global, positive_key=positive_global,
                                          negative_keys=negative_global)
                l_cl_label_avg += 0.5 * l_cl_p + 0.5 * l_cl_g
            l_cl_label_avg = l_cl_label_avg / label_cnt
            l_cl_batch_avg += l_cl_label_avg
        l_cl_batch_avg /= B

        return l_cl_batch_avg

    def _calculate_contrastive_loss_point_segment_active_sampling(self):
        """
        联合使用点级和全局级损失对比损失，并引入负样本主动学习采样
        注意设置InfoNCELoss为paired模式
        """

        B, N, D = self.output.shape
        l_cl_batch_avg = 0
        for output, label in zip(self.output, self.label):  # for batch
            label_unique, counts = torch.unique(label, return_counts=True)  # 所有label种类
            label_cnt = 0
            l_cl_label_avg = 0

            features_of_label = {}  # 存储每个label的特征
            features_of_label_mean = {}  # 存储每个label的平均特征
            for label_elem, count in zip(label_unique, counts):
                # 过滤样本数少于2的标签的样本
                if count < 2:
                    continue
                label_cnt += 1
                feature = output[label == label_elem]  # (N1, D)  N1为该类的脉冲总数
                features_of_label[label_elem.item()] = feature
                features_of_label_mean[label_elem.item()] = feature.mean(dim=0, keepdim=True)

            for label_elem, count in zip(label_unique, counts):
                if count < 2:
                    continue
                mask = (label == label_elem)
                feature = features_of_label[label_elem.item()]  # 取出该label的特征
                shuffle_idx = torch.randperm(feature.shape[0])

                ### 点级 ###
                mid = len(shuffle_idx) // 2
                anchor_point_idx, positive_point_idx = shuffle_idx[:mid], shuffle_idx[mid:2 * mid]  # 一半为锚点，一半为正样本
                anchor_point = feature[anchor_point_idx, :]
                positive_point = feature[positive_point_idx, :]

                negative_point_pool = output[~mask, :]  # 所有负样本

                select_type = 'entropy_based'  # 负样本选择策略
                neg_num = 50                   # 负样本选择数量

                negative_point = []
                for anchor, positive in zip(anchor_point, positive_point):
                    similarities = F.cosine_similarity(anchor, negative_point_pool, dim=-1)   # 计算锚点与负样本相似度
                    if select_type == 'sim_based':
                        _, indices = torch.topk(similarities, k=min(neg_num, len(negative_point_pool)))

                    elif select_type == 'entropy_based':
                        positive_similarity = F.cosine_similarity(anchor, positive, dim=-1)
                        uncertainties = []  # 计算包含当前负样本后，相似度的熵，寻找使熵最大的前50个样本
                        for neg_sim in similarities:
                            sims = torch.cat([positive_similarity.unsqueeze(0), neg_sim.unsqueeze(0)])
                            probs = F.softmax(sims / 0.1, dim=0)
                            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8))
                            uncertainties.append(uncertainty)
                        uncertainties = torch.tensor(uncertainties)
                        # 选择不确定性最高的样本
                        _, indices = torch.topk(uncertainties, k=min(neg_num, len(negative_point_pool)))
                    else:
                        raise ValueError('select_type must be "sim_based" or "entropy_based"')
                    negative_points_for_anchor = negative_point_pool[indices]
                    negative_point.append(negative_points_for_anchor)
                negative_point = torch.tensor(negative_point).unsqueeze(0)               # (1, neg_num, D)

                ### 片段级 ###
                anchor_global = feature[shuffle_idx[0], :].unsqueeze(0)  # 随机取出一个为锚点  (1, D)
                positive_global = features_of_label_mean[label_elem.item()]  # 该类的平均特征为正样本
                negative_global = torch.cat(
                    [value for key, value in features_of_label_mean.items() if key != label_elem],
                    dim=0)  # 其他类的平均特征为负样本

                l_cl_p = self.cri_infonce(query=anchor_point, positive_key=positive_point, negative_keys=negative_point)
                l_cl_g = self.cri_infonce(query=anchor_global, positive_key=positive_global,
                                          negative_keys=negative_global)
                l_cl_label_avg += 0.5 * l_cl_p + 0.5 * l_cl_g
            l_cl_label_avg = l_cl_label_avg / label_cnt
            l_cl_batch_avg += l_cl_label_avg
        l_cl_batch_avg /= B

        return l_cl_batch_avg
