import scanpy as sc
import anndata as ad
import numpy as np
from numba import njit
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix
import scipy.stats as stats
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
from skimage.measure import block_reduce
import cv2
# import os
# import sys
# import time
import torch
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision import transforms, datasets
import itertools
from collections import deque
from entmax import entmax_bisect
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

from .util import adjust_learning_rate
from .util import pad_to_divisible
from .model import SILPAG_model, AnchorPool
from .agot import get_final_transport_plan, transform_distributions, OT_EM


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train1epoch_stage1(epoch, model, optimizer, train_loader, hist_data, args):
    train_loss1 = train_loss2 = train_loss3 = train_loss4 = train_loss5 = train_loss6 = train_loss7 = 0
    model.train()
    model.zero_code_usage()
    for view, hk_idx, _ in train_loader:
        bsz = view[0].size(0)
        view = [i.to(args.device) for i in view]
        if epoch <= args.warmup_epochs:
            z, ranking_loss, huber_loss = model.forward_wo_vq(view, hist_data)

            cosine_loss = torch.zeros_like(ranking_loss.sum())
            for e in z:
                cosine_loss = cosine_loss + cosine_regularization(e)

            train_loss1 += bsz * (ranking_loss.sum().item())
            train_loss3 += bsz * (huber_loss.sum().item())
            train_loss7 += bsz * (cosine_loss.sum().item())

            total_loss = 2 * ranking_loss.sum() + 1 * huber_loss.sum() + 2 * cosine_loss.sum()
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        else:
            z, _, ranking_loss, huber_loss, vq_loss = model.forward_marker(view, hist_data)

            if args.contrastive:
                calculate_clrloss = CrossViewLoss(batch_size=len(z[0]), device=args.device)
                cosine_loss = calculate_clrloss(z[0], z[1])
            else:
                cosine_loss = torch.zeros_like(ranking_loss.sum())
                for e in z:
                    cosine_loss = cosine_loss + cosine_regularization(e)

            
                
            train_loss1 += bsz * (ranking_loss.sum().item())
            # train_loss2 += bsz * contras_loss.item()
            train_loss3 += bsz * (huber_loss.sum().item())
            train_loss6 += bsz * (vq_loss.sum().item())
            train_loss7 += bsz * (cosine_loss.sum().item())

            # total_loss1 = ranking_loss.sum() + args.alpha * sparse_loss.sum() + \
            #                 2 * value_loss.sum() + args.beta * contras_loss + vq_loss.sum()
            total_loss = 2 * ranking_loss.sum() + 1 * huber_loss.sum() + 10 * vq_loss.sum() + 5 * cosine_loss.sum()
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    if args.trace: 
        if epoch % int(args.pre_epochs / 5) == 0:
            print('Ranking loss: ', train_loss1 / len(train_loader.dataset))
            print('Huber loss: ', train_loss3 / len(train_loader.dataset))
            # print('Multi-view loss: ', train_loss2 / len(train_loader.dataset))
            print('Cosine loss: ', train_loss7 / len(train_loader.dataset))
            print('VQ loss: ', train_loss6 / len(train_loader.dataset))


def train1epoch_stage2(epoch, model, optimizer, anchorpool, train_loader, hist_data, marker_loader, Q, PI, P0, KAPPA, args):
    # torch.autograd.set_detect_anomaly(True)
    train_loss1 = train_loss2 = train_loss3 = train_loss4 = train_loss5 = train_loss6 = train_loss7 = 0
    model.train()
    model.zero_code_usage()
    d = torch.full((len(train_loader.dataset),), 0, dtype=torch.float, device=args.device, requires_grad=False)
    for (view, hk_idx, idx), (marker_view, _) in zip(train_loader, itertools.cycle(marker_loader)):
        bsz = view[0].size(0)
        view = [i.to(args.device) for i in view]
        marker_view = [i.to(args.device) for i in marker_view]
            
        z, s, ranking_loss, huber_loss, vq_loss = model(view, hist_data)

        if args.contrastive:
                calculate_clrloss = CrossViewLoss(batch_size=len(z[0]), device=args.device)
                cosine_loss = calculate_clrloss(z[0], z[1])
        else:
            cosine_loss = torch.zeros_like(ranking_loss.sum())
            for e in z:
                cosine_loss = cosine_loss + cosine_regularization(e)

        train_loss1 += bsz * (ranking_loss.sum().item())
        train_loss3 += bsz * (huber_loss.sum().item())
        train_loss6 += bsz * (vq_loss.sum().item())
        train_loss7 += bsz * (cosine_loss.sum().item())

        total_loss1 = 2 * ranking_loss.sum() + 1 * huber_loss.sum() + 10 * vq_loss.sum() + 5 * cosine_loss.sum()
        
        # Anchor-gene-Guided Optimal Transport
        anchor_mask = hk_idx == 1
        
        if epoch > 1:
            for i in args.ref_index:
                for j in range(args.num_slice):
                    if j == i:
                        continue
                    a, q_a = anchorpool.get()
                    s_i = torch.cat([s[i], a[i].to(args.device)], dim=0)
                    s_j = torch.cat([s[j], a[j].to(args.device)], dim=0)
                    anchor_mask_all = torch.cat([anchor_mask, torch.full((a[i].shape[0],), True)], dim=0)

                    q_g = torch.cat([Q[idx], q_a.to(args.device)], dim=0)

                    AGOT_loss, q_g, d_g, p0, kappa, pi = OT_EM(s_i, s_j, anchor_mask_all, bsz, len(train_loader.dataset), 
                                                               model.get_codebook(i), model.get_codebook(j), PI, q_g, P0, 
                                                               KAPPA, device=args.device)
                    P0 = p0
                    Q[idx] = q_g[:bsz]
                    PI = pi
                    KAPPA = kappa
                    d[idx] = d_g[:bsz]

                    s_anchor = [si[anchor_mask] for si in s]
                    _, _ = anchorpool(s_anchor, Q[idx][anchor_mask])
                    train_loss2 += bsz * AGOT_loss.item()
                    total_loss1 += args.beta * AGOT_loss.sum()

        else:
            s_hk = [i[anchor_mask] for i in s]
            _, _ = anchorpool(s_hk)

        _, _, ranking_loss2, huber_loss2, vq_loss2 = model.forward_marker(marker_view, hist_data)

        total_loss = total_loss1 + ranking_loss2.sum() + huber_loss2.sum() + vq_loss2.sum()
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

    if args.trace: 
        if epoch % int(args.epochs / 5) == 0:
            print('Ranking loss: ', train_loss1 / len(train_loader.dataset))
            print('Huber loss: ', train_loss3 / len(train_loader.dataset))
            print('AGOT loss: ', train_loss2 / len(train_loader.dataset))
            print('VQ loss: ', train_loss6 / len(train_loader.dataset))
            print('Anchor Pool size: ', train_loader.dataset.labels.sum())
            print('P0: ', P0)
            print('Mean Q: ', Q.mean())

    if epoch > 1:
        return Q, d, PI, P0, KAPPA, train_loss1 / len(train_loader.dataset), train_loss3 / len(train_loader.dataset)


def findmarker(z, n_neighbors=15, top_n_per_cluster=50, resolution=1.0):
    n_genes = z[0].shape[0]

    idx, label = [], []
    for i in range(len(z)):
        adata = sc.AnnData(z[i])
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
        sc.tl.leiden(adata, resolution=resolution)
        labels = adata.obs['leiden'].to_numpy().astype(int)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        top_indices_all_clusters = []
        top_labels_list = []
        for cluster_id in unique_labels:

            indices_in_cluster = np.where(labels == cluster_id)[0]
            if len(indices_in_cluster) == 0:
                continue

            if len(indices_in_cluster) < top_n_per_cluster:

                upsampled_indices = np.random.choice(
                    indices_in_cluster,
                    size=top_n_per_cluster,
                    replace=True
                )
                top_indices_all_clusters.extend(upsampled_indices)

            else:

                embeddings_in_cluster = z[i][indices_in_cluster]
                centroid = embeddings_in_cluster.mean(axis=0)
                distances_sq = np.sum((embeddings_in_cluster - centroid)**2, axis=1)
                closest_local_indices = np.argsort(distances_sq)[:top_n_per_cluster]
                original_indices = indices_in_cluster[closest_local_indices]
                top_indices_all_clusters.extend(original_indices)

            top_labels_list.extend([cluster_id] * top_n_per_cluster)

        idx.append(top_indices_all_clusters)
        label.append(top_labels_list)
    return idx, label


def findmarker_kmeans(z, top_n_per_cluster=50, n_clusters=64):
    idx, label = [], []
    for i in range(len(z)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(z[i])
        
        unique_labels = np.unique(labels)
        
        top_indices_all_clusters = []
        top_labels_list = []
        for cluster_id in unique_labels:
            
            indices_in_cluster = np.where(labels == cluster_id)[0]
            if len(indices_in_cluster) == 0:
                continue
            
            if len(indices_in_cluster) < top_n_per_cluster:
                upsampled_indices = np.random.choice(
                    indices_in_cluster,
                    size=top_n_per_cluster,
                    replace=True
                )
                top_indices_all_clusters.extend(upsampled_indices)
            
            else:
                embeddings_in_cluster = z[i][indices_in_cluster]
                centroid = embeddings_in_cluster.mean(axis=0)
                distances_sq = np.sum((embeddings_in_cluster - centroid)**2, axis=1)
                closest_local_indices = np.argsort(distances_sq)[:top_n_per_cluster]
                original_indices = indices_in_cluster[closest_local_indices]
                top_indices_all_clusters.extend(original_indices)
                
            top_labels_list.extend([cluster_id] * top_n_per_cluster)
            
        idx.append(top_indices_all_clusters)
        label.append(top_labels_list)
        
    return idx, label


def train_marker(train_loader, hist_data=None, args=None):
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)
    print('Shape of each View: ', args.img_size)

    if args.train:
         # Set the model
        model = SILPAG_model(args)
        model = model.to(args.device)
            
        optimizer = set_optimizer(model, args)
        
        if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
        else: hist_data = [0 for i in range(args.num_slice)]
        
        # ------------------------------ Satge I: Pretrain ------------------------------
        for epoch in tqdm(range(1, args.pre_epochs + 1), desc='Pre-training', unit='epochs'):
            train1epoch_stage1(epoch, model, optimizer, train_loader, hist_data, args)
        
        # model.Visualize_codeusage()

        # ------------------------------ Dropout the useless code ------------------------------
        if args.pre_epochs>0:
            test_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            dataset = train_loader.dataset
            z = get_embedding(model, dataset, hist_data, args)
            # idx, marker_label = findmarker(z, n_neighbors=15, top_n_per_cluster=30, resolution=2)
            idx, marker_label = findmarker_kmeans(z, top_n_per_cluster=1, n_clusters=50)
            marker_data = []
            for i in range(args.num_slice):
                data = dataset.gene_images[i][idx[i]]
                marker_data.append(data)
            marker_dataset = GeneDataset_marker(marker_data, None)
            marker_loader = DataLoader(marker_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
            
            model.zero_code_usage()
            with torch.no_grad():
                for view, _ in marker_loader:
                    view = [i.to(args.device) for i in view]
                    _ = model.forward_marker(view, hist_data)
            model.drop_useless_code()
        args = model.return_args()

        # ------------------------------ Staget II: Train ------------------------------     
        # initialize
        P0 = train_loader.dataset.labels.mean()
        anchor = AnchorPool(P0, args)

        Q = torch.full((len(train_loader.dataset),), P0, device=args.device, requires_grad=False)
        R = Q
        KAPPA = None
        PI = torch.tensor(0.)
        codebook_size = []
        for i in range(args.num_slice):
            codebook_size.append(model.get_codebook(i).shape[0])
        for i in args.ref_index:
            for j in range(args.num_slice):
                if j == i:
                    continue
                K_A, K_B = codebook_size[i], codebook_size[j]
                PI = (torch.ones(K_A, K_B).to(args.device) / (K_A * K_B))
                

        H = torch.zeros(len(train_loader.dataset), 1, requires_grad=False, device=args.device) 
        # L1, L2 = [], []
        for epoch in tqdm(range(1, args.epochs + 1), desc='Training', unit='epochs'):
            adjust_learning_rate(epoch, args, optimizer)
            # hk_num.append(train_loader.dataset.labels.sum())
            if epoch > 1:
                # Q, d, PI, P0, KAPPA = train1epoch_stage2(epoch, model, optimizer, anchor, train_loader, hist_data, marker_loader, Q, PI, P0, KAPPA, args)
                Q, d, PI, P0, KAPPA, loss1, loss2 = train1epoch_stage2(epoch, model, optimizer, anchor, train_loader, hist_data, marker_loader, Q, PI, P0, KAPPA, args)
                # L1.append(loss1)
                # L2.append(loss2)



                if args.num_slice > 1:

                    with torch.no_grad():
                        code_idx = get_code_idx(model, train_loader.dataset, hist_data, args)
                        s1, s2 = torch.tensor(code_idx[0], device=args.device), torch.tensor(code_idx[1], device=args.device)
                        # print(s1.max(), s2.max())
                        _, Q, d, P0, KAPPA, PI = OT_EM(s1, s2, torch.tensor(train_loader.dataset.labels), 
                                                                   len(train_loader.dataset), len(train_loader.dataset), 
                                                                    model.get_codebook(0), model.get_codebook(1), PI, Q, P0, 
                                                                    KAPPA, device=args.device)
                        
                    # if epoch % int(args.epochs / 5) == 0:
                    #     plot_PI(PI)

                    H = torch.cat((H, Q.unsqueeze(1)), dim=1)

                    beta_R = 0.7
                    R = beta_R * R + (1 - beta_R) * Q

                    if epoch > 10 and H.shape[1] > 4:
                        
                        subH = H[:, -3:]

                        tau_up, tau_down = torch.quantile(Q, 0.85), torch.quantile(Q, 0.65)
                        
                        up1 = R > tau_up
                        up2 = torch.all(subH > tau_up, dim=1)
                        up = up1 & up2

                        down1 = R < tau_down
                        down2 = d >= torch.quantile(d, 0.6)
                        down = down1 & down2

                        if torch.any(up): 
                            train_loader.dataset.labels[up.cpu()] = 1
                        if torch.any(down): 
                            train_loader.dataset.labels[down.cpu()] = 0

            else: train1epoch_stage2(epoch, model, optimizer, anchor, train_loader, hist_data, marker_loader, Q, PI, P0, KAPPA, args)
            
        if args.save:
            torch.save(model.state_dict(), f"{args.model_path + '.pkl'}")
            model.args.PI = PI.data.cpu().tolist()
            model.args.save_config()
    else:
        args = args.load_config(args.model_path)
        model = SILPAG_model(args)
        model = model.to(args.device)
        model.load_state_dict(torch.load(f"{args.model_path + '.pkl'}"))
        PI = args.PI
        Q = None
        # L1, L2 = None, None

    return model, train_loader.dataset, PI, Q # , L1, L2


def plot_PI(PI):
    data_np = PI.cpu().numpy()

    annotations = np.full(data_np.shape, "", dtype=object)

    max_in_rows = data_np.max(axis=1, keepdims=True)
    is_max_in_row = (data_np == max_in_rows)

    max_in_cols = data_np.max(axis=0, keepdims=True)
    is_max_in_col = (data_np == max_in_cols)

    row_max_indices = np.argmax(data_np, axis=1)
    for i, j in enumerate(row_max_indices):
        annotations[i, j] = f"{data_np[i, j]:.2f}"

    col_max_indices = np.argmax(data_np, axis=0)
    for j, i in enumerate(col_max_indices):
        annotations[i, j] = f"{data_np[i, j]:.2f}"

    plt.figure(figsize=(3, 2.5))

    xticklabels = [f'Tgt_code {i+1}' for i in range(data_np.shape[1])]
    yticklabels = [f'Ref_code {i+1}' for i in range(data_np.shape[0])] 
    ax = sns.heatmap(
        data_np,
        annot=annotations,   
        fmt="",              
        cmap='RdBu_r',
        linewidths=.5,
        xticklabels=xticklabels,
        yticklabels=yticklabels
    )
    plt.title('Ref-Tgt Codebook Transport Plan', fontsize=8)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=8)

    plt.tight_layout()
    plt.show()


def cosine_regularization(embeddings):

    batch_size = embeddings.size(0)
    num_off_diagonal_elements = batch_size * (batch_size - 1)
    if num_off_diagonal_elements == 0:
        return torch.tensor(0.0, device=embeddings.device)

    normalized_embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    cosine_sim_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.t())

    off_diagonal_squared_sum = torch.sum(cosine_sim_matrix**2) - batch_size
    
    loss = off_diagonal_squared_sum / num_off_diagonal_elements
    
    return loss


class GeneDataset(torch.utils.data.Dataset):
    def __init__(self, gene_images, labels, args, use_filter=False):
        self.num_slice = len(gene_images)
        args.orig_img_size = [i.shape[1:] for i in gene_images]
        
        self.gene_images = gene_images
        for i in range(self.num_slice):
            if args.resize_factor[i] > 1:
                self.gene_images[i] = block_reduce(self.gene_images[i], block_size=(1, args.resize_factor[i], args.resize_factor[i]), func=np.sum)
                tgt_shape = (self.gene_images[i].shape[0], self.gene_images[i].shape[1], self.gene_images[i].shape[2])
                args.resized_img_size.append(tgt_shape)
            else: 
                args.resized_img_size.append(self.gene_images[i].shape)
            self.gene_images[i] = pad_to_divisible(self.gene_images[i], patch_size=args.patch_size[i][0])
            if use_filter:
                self.gene_images[i] = np.array([gaussian_filter(img, sigma=1) for img in self.gene_images[i]])
        
        args.img_size = [i.shape[1:] for i in self.gene_images]   

        if labels is not None: 
            self.labels = labels
        else: 
            self.labels = np.zeros(self.__len__())


    def __len__(self):
        return max([i.shape[0] for i in self.gene_images])

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_slice):
            img = torch.tensor(self.gene_images[i][int(idx % self.gene_images[i].shape[0])], dtype=torch.float32)
            img = img.unsqueeze(0)
            data.append(img)
        label = torch.tensor(self.labels[idx]).long()
        return data, label, idx


class GeneDataset_marker(torch.utils.data.Dataset):
    def __init__(self, gene_images, labels):
        self.num_slice = len(gene_images)
        
        self.gene_images = gene_images  

        if labels is not None: 
            self.labels = labels
        else: 
            self.labels = np.zeros(self.__len__())


    def __len__(self):
        return max([i.shape[0] for i in self.gene_images])

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_slice):
            img = torch.tensor(self.gene_images[i][int(idx % self.gene_images[i].shape[0])]).float()
            img = img.unsqueeze(0)
            data.append(img)
        label = torch.tensor(self.labels[idx]).long()
        return data, label


def set_optimizer(model, args):
    # return optimizer
    if args.optimizer=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer=='RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer


@torch.no_grad()
def get_embedding(model, dataset, hist_data=None, args=None, normalize=False):

    if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
    else: hist_data = [0 for i in range(args.num_slice)]

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    embed = []
    for i in range(args.num_slice):
        embed.append([])
    
    model.eval()
    model.zero_code_usage()
    with torch.no_grad():
        for view, hk_idx, _ in test_loader:
            view = [i.float().to(args.device) for i in view]
            z, _, _, _, _ = model(view, hist_data)
            for i in range(args.num_slice):
                embed[i].append(z[i])

    for i in range(args.num_slice):
        embed[i] = torch.cat(embed[i], dim=0)
        embed[i] = embed[i].data.cpu().numpy()
    if normalize:
        scaler = StandardScaler()
        for i in range(args.num_slice):
            embed[i] = scaler.fit_transform(embed[i])
            
    return embed


@torch.no_grad()
def get_code_idx(model, dataset, hist_data=None, args=None):

    if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
    else: hist_data = [0 for i in range(args.num_slice)]

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    code_idx = []
    for i in range(args.num_slice):
        code_idx.append([])
    
    model.eval()
    model.zero_code_usage()
    with torch.no_grad():
        for view, hk_idx, _ in test_loader:
            view = [i.float().to(args.device) for i in view]
            _, s, _, _, _ = model(view, hist_data)
            
            for i in range(args.num_slice):

                # z[i] = model.forward_entmax(z[i], i)

                code_idx[i].append(s[i])

    for i in range(args.num_slice):
        code_idx[i] = torch.cat(code_idx[i], dim=0)
        code_idx[i] = code_idx[i].data.cpu().numpy()
            
    return code_idx


@torch.no_grad()
def get_code(model, dataset, hist_data=None, pi_init=None, source_idx=0, target_idx=1, args=None):

    if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
    else: hist_data = [0. for i in range(args.num_slice)]

    code_idx = get_code_idx(model, dataset, hist_data, args)
    s1, s2 = torch.tensor(code_idx[source_idx], device=args.device), torch.tensor(code_idx[target_idx], device=args.device)
    codebook1, codebook2 = model.get_codebook(source_idx), model.get_codebook(target_idx)
    with torch.no_grad():
        q_g = torch.full((len(dataset),), dataset.labels.mean(), device=args.device, requires_grad=False)
        anchor_mask = torch.tensor(dataset.labels == 1)
        pi, u, v = get_final_transport_plan(q_g, s1, s2, codebook1, codebook2, anchor_mask, args.device, pi_init, gw_max_iter=1000, inner_sink_iter=100, gw_epsilon=1e-3)
        # pi = pi_init
        s2_hat, s1_hat = transform_distributions(s1, s2, pi)
        code = [model.get_genecode(s2_hat, target_idx).data.cpu().numpy(), model.get_genecode(s2, target_idx).data.cpu().numpy()]
    return code


@torch.no_grad()
def get_both(model, dataset, hist_data=None, pi_init=None, source_idx=0, target_idx=1, args=None):

    if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
    else: hist_data = [0. for i in range(args.num_slice)]

    code_idx = get_code_idx(model, dataset, hist_data, args)
    s1, s2 = torch.tensor(code_idx[source_idx], device=args.device), torch.tensor(code_idx[target_idx], device=args.device)
    codebook1, codebook2 = model.get_codebook(source_idx), model.get_codebook(target_idx)
    with torch.no_grad():
        q_g = torch.full((len(dataset),), dataset.labels.mean(), device=args.device, requires_grad=False)
        anchor_mask = torch.tensor(dataset.labels == 1)
        # pi, u, v = get_final_transport_plan(q_g, s1, s2, codebook1, codebook2, anchor_mask, args.device, pi_init, gw_max_iter=1000, inner_sink_iter=100, gw_epsilon=1e-3)
        pi = pi_init
        s2_hat, s1_hat = transform_distributions(s1, s2, pi)
        codeidx1, code1 = model.get_both(s2_hat, target_idx)
        codeidx2, code2 = model.get_both(s2, target_idx)
        code = [code1.data.cpu().numpy(), code2.data.cpu().numpy()]
        code_idx = [codeidx1.data.cpu().numpy(), codeidx2.data.cpu().numpy()]
    return code_idx, code


@torch.no_grad()
def get_code_single(model, dataset, hist_data=None, tgt_idx=0, args=None):
    code_idx = get_code_idx(model, dataset, hist_data, args)
    s = torch.tensor(code_idx[tgt_idx], device=args.device)
    # codebook = model.get_codebook(tgt_idx)
    with torch.no_grad():
        code = [model.get_genecode(s, tgt_idx).data.cpu().numpy()]
    return code


@torch.no_grad()
def get_code_distance(model, dataset, hist_data=None, args=None):

    if hist_data is not None:
            hist_data = [torch.tensor(i, device=args.device).float() for i in hist_data]
    else: hist_data = [0. for i in range(args.num_slice)]

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    d = []
    for i in range(args.num_slice):
        d.append([])
    
    model.eval()
    model.zero_code_usage()
    with torch.no_grad():
        for view, hk_idx in test_loader:
            view = [i.float().to(args.device) for i in view]
            _, z, _, _, _ = model(view, hist_data)
            
            for i in range(args.num_slice):

                d[i].append(z[i])

    for i in range(args.num_slice):
        d[i] = torch.cat(d[i], dim=0)
        d[i] = d[i].data.cpu().numpy()
            
    return d


def cal_scores(all_view_encoded, num_neibs):
    from sklearn.neighbors import NearestNeighbors
    num_views = len(all_view_encoded)
    num_obj = all_view_encoded[0].shape[0]

    encoded_concen = np.concatenate(all_view_encoded,axis=1)
    NNK = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree')
    
    nbrs = NNK.fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)   
    knn_scores = np.sum(distances, axis=1)
    tmp = list(indices[0][1:])
    
    knn_scores = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))
    return knn_scores


def cal_scores2(all_view_encoded, num_neibs):
    from sklearn.neighbors import NearestNeighbors
    num_views = len(all_view_encoded)
    num_obj = all_view_encoded[0].shape[0]

    encoded_concen = np.concatenate(all_view_encoded,axis=1)
    NNK = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree')
    
    nbrs = NNK.fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)   
    knn_scores = np.sum(distances, axis=1)
    tmp = list(indices[0][1:])
    
    neib_score = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))

    distance = np.sqrt(np.sum((all_view_encoded[0] - all_view_encoded[1]) ** 2, axis=1))
    min_distance = np.min(distance)
    max_distance = np.max(distance)
    view_score = (distance - min_distance) / (max_distance - min_distance)
    return (neib_score + view_score) / 2


def cal_scores3(all_view_encoded, num_neibs):
    from sklearn.neighbors import NearestNeighbors
    num_views = len(all_view_encoded)
    num_obj = all_view_encoded[0].shape[0]
    
    # Verify all views have the same embedding dimension
    dims = [data.shape[1] for data in all_view_encoded]
    if len(set(dims)) > 1:
        raise ValueError("All views must have the same embedding dimension for VIC calculation.")
    dim = dims[0]
    
    # Step 1: Compute overall embedding and get KNN indices
    encoded_concen = np.concatenate(all_view_encoded, axis=1)
    nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(encoded_concen)
    _, overall_indices = nbrs.kneighbors(encoded_concen)
    overall_indices = overall_indices[:, 1:]  # Exclude self
    
    # Step 2: Precompute within-view KNN indices
    view_within_indices = []
    for v in range(num_views):
        data_view = all_view_encoded[v]
        nbrs_view = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(data_view)
        _, indices_view = nbrs_view.kneighbors(data_view)
        view_within_indices.append(indices_view[:, 1:])  # Exclude self
    
    # Step 3: Compute NIC scores per view (vectorized)
    NIC_scores = np.zeros(num_obj)
    for v in range(num_views):
        data_view = all_view_encoded[v]
        # Term1: Squared distances to neighbors from overall KNN
        neighbor_data = data_view[overall_indices]  # (num_obj, num_neibs, dim)
        diff1 = data_view[:, np.newaxis, :] - neighbor_data
        term1 = np.sum(diff1**2, axis=(1, 2))  # Sum over neighbors and dimensions
        
        # Term2: Squared distances to neighbors within same view
        neighbor_data = data_view[view_within_indices[v]]  # (num_obj, num_neibs, dim)
        diff2 = data_view[:, np.newaxis, :] - neighbor_data
        term2 = np.sum(diff2**2, axis=(1, 2))  # Sum over neighbors and dimensions
        
        NIC_scores += (term1 - term2)
    
    # Step 4: Compute VIC scores (vectorized over all objects and view pairs)
    all_emb = np.stack(all_view_encoded, axis=1)  # (num_obj, num_views, dim)
    diff = all_emb[:, :, np.newaxis, :] - all_emb[:, np.newaxis, :, :]
    sq_diff = diff**2
    sq_dist = np.sum(sq_diff, axis=-1)  # (num_obj, num_views, num_views)
    
    # Upper triangular mask (without diagonal)
    mask = np.triu(np.ones((num_views, num_views)), k=1).astype(bool)
    VIC_scores = np.sum(sq_dist * mask[np.newaxis, :, :], axis=(1, 2))
    
    # Step 5: Min-Max Normalization
    min_nic, max_nic = np.min(NIC_scores), np.max(NIC_scores)
    min_vic, max_vic = np.min(VIC_scores), np.max(VIC_scores)
    
    range_nic = max_nic - min_nic
    range_vic = max_vic - min_vic
    if range_nic == 0:
        range_nic = 1e-8
    if range_vic == 0:
        range_vic = 1e-8
    
    nic_normalized = (NIC_scores - min_nic) / range_nic
    vic_normalized = (VIC_scores - min_vic) / range_vic
    
    # Step 6: Average normalized scores
    total_score = (nic_normalized + vic_normalized) / 2
    return total_score


@njit
def fill_gene_matrix(shape, array_y, array_x, g_values):
    g_matrix = np.zeros(shape=shape)
    for i in range(len(array_y)):
        row_ix = array_y[i]
        col_ix = array_x[i]
        g_matrix[row_ix, col_ix] = g_values[i]
    return g_matrix


def data_process(adata_raw):
    """
    process adata and return indicator
    """
    from scipy.sparse import csr_matrix
    
    adata_raw.obs['array_x'] = np.ceil((adata_raw.obs['array_col'] - adata_raw.obs['array_col'].min())).astype(int)
    adata_raw.obs['array_y'] = adata_raw.obs['array_row'] - adata_raw.obs['array_row'].min()
    adata = adata_raw.copy()
    all_genes = adata.var.index.values
    shape = (adata.obs['array_y'].max() + 1, adata.obs['array_x'].max() + 1)
    print('Size of gene image: ', shape)
    indicator = fill_gene_matrix(shape, adata.obs['array_y'].values, adata.obs['array_x'].values, np.ones(adata.shape[0]))
    
    adata.X = adata.X.todense() if isspmatrix(adata.X) else adata.X
    all_gene_exp_matrices = {}
    
    for gene in tqdm(all_genes, desc="adata2image", unit="gene"):
        g_values = np.array(adata[:, gene].X).flatten()
        g_matrix = fill_gene_matrix(shape, adata.obs['array_y'].values, 
                                    adata.obs['array_x'].values, g_values)
        all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
    all_gmat = {k: all_gene_exp_matrices[k] for k in list(all_gene_exp_matrices.keys())}
    dataset = np.array(list(all_gmat.values()))
    dataset = np.array([x.todense() for x in dataset])
    adata.X = csr_matrix(np.array(adata.X))
    return dataset, indicator, shape


def histology_to_image(adata_raw, histology_features):
    
    @njit
    def fill_feature_image_C_H_W(shape_2d, array_y, array_x, histology_features):
        n_spots = len(array_y)
        n_features = histology_features.shape[1]
        feature_image = np.zeros(shape=(n_features, shape_2d[0], shape_2d[1]))
        
        for i in range(n_spots):
            row_ix = array_y[i]
            col_ix = array_x[i]
            feature_image[:, row_ix, col_ix] = histology_features[i, :]
                
        return feature_image

    
    if not isinstance(histology_features, np.ndarray):
        histology_features = np.array(histology_features)
        
    n_features = histology_features.shape[1]
    
    array_x = np.ceil((adata_raw.obs['array_col'] - adata_raw.obs['array_col'].min())).astype(int).values
    array_y = (adata_raw.obs['array_row'] - adata_raw.obs['array_row'].min()).astype(int).values

    shape_2d = (array_y.max() + 1, array_x.max() + 1)

    feature_image = fill_feature_image_C_H_W(shape_2d, array_y, array_x, histology_features)

    return feature_image


class Config:
    def __init__(self):
        self.trace = True
        self.device = 'cuda'
        self.hist = [False, False]
        self.distr = ['nb', 'nb']
        self.contrastive = True
        
        self.alpha = 3e-1
        self.beta = 1
        # self.gamma = 2

        self.num_slice = 2
        self.ref_index = [0]
        self.resize_factor = [1, 1]
        self.orig_img_size = []
        self.resized_img_size = []
        self.img_size = []
        self.patch_size = []
        self.K = 1024
        self.anchor_size = 1024
        
        self.print_freq = 10  # print frequency
        self.batch_size = 128  # batch_size
        self.pre_epochs = 50
        self.epochs = 200  # number of training epochs
        # self.decoder_epochs = 10
        self.warmup_epochs = 50
        self.hkqueue_size = 128
        self.anchor_epochs = 10

        # optimization
        self.optimizer = 'AdamW'
        self.learning_rate = 1.5e-3  # learning rate
        self.lr_decay_epochs = [200]  # where to decay lr, can be a list
        self.lr_decay_rate = 0.1  # decay rate for learning rate
        self.weight_decay = 1e-3  # weight decay
        self.momentum = 0.9  # momentum

        self.train = True
        self.save = True

        # model definition
        self.model = 'MAE'
        self.dropout_rate=0.3
        self.feat_dim = 64  # dim of feat for inner product
        self.vq_dim = 16

        self.PI = []

        self.model_path = './save_model/model'  # path to save model

    def parse(self):
        return self

    def save_config(self):
        import json
        import os
        try:
            file_path = self.model_path+'.json'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(vars(self), f, indent=4)
        except Exception as e:
            print(f"Error in saving Config: {e}")

    @classmethod
    def load_config(cls, file_path):
        import json
        import os
        try:
            file_path = file_path+'.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            instance = cls()
            
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            try:
                instance.PI = torch.tensor(instance.PI, device=instance.device).view(instance.K[0], instance.K[1])
            except Exception as pi_error:
                print("Cannot load transport plan PI.")
            return instance
        
        except FileNotFoundError:
            print(f"Cannot find the config file {file_path}")
            return None
        except Exception as e:
            print(f"Error in loading Config: {e}")
            return None
        

class CrossViewLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        mask = self.mask_correlated_samples(self.batch_size)
        self.mask = mask.to(device)

    def mask_correlated_samples(self, N):
        mask = torch.ones((2*N, 2*N))
        mask = mask.fill_diagonal_(0)
        # for i in range(N):
        #     mask[i, N + i] = 0
        #     mask[N + i, i] = 0
        mask[:N, :N] = 0
        mask[N:, N:] = 0
        mask = mask.bool()
        return mask
        
    def forward(self, emb_i, emb_j): # emb_i, emb_j 
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
    
        # Cross-view positive pairs
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / (0.5 * torch.sum(denominator, dim=1)))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss