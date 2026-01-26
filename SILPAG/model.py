import numpy as np
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Gamma, Exponential
from einops import rearrange
from tqdm import tqdm
from entmax import entmax_bisect

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .util import unpad_to_original, sinkhorn


    
class AnchorPool(nn.Module):
    def __init__(self, prob, args):
        super().__init__()
        self.size = args.anchor_size
        self.dims = args.K
        self.num_slice = args.num_slice
        self.prob = prob
        
        pools = []
        for dim in self.dims:
            pool_tensor = torch.randn(self.size, dim)
            pools.append(nn.Parameter(pool_tensor, requires_grad=False))
        self.anchor_pools = nn.ParameterList(pools)
        
        prob_tensor = torch.ones(self.size) * self.prob
        self.anchor_probs = nn.Parameter(prob_tensor, requires_grad=False)
        
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_list: list[torch.Tensor], probs: torch.Tensor = None):

        assert keys_list and isinstance(keys_list, list), "keys_list should be a list"
        batch_size = keys_list[0].shape[0]

        # --- 修改：使用共享指针 ---
        ptr = int(self.ptr)
        end_idx = ptr + batch_size
        
        if probs is None:
            new_probs = torch.ones(batch_size, device=keys_list[0].device, dtype=keys_list[0].dtype) * self.prob
        else:
            assert probs.shape == (batch_size,), f"probs shape should be ({batch_size},), but got {probs.shape}"
            new_probs = probs

        if end_idx <= self.size:
            self.anchor_probs[ptr:end_idx] = new_probs
            for i in range(self.num_slice):
                self.anchor_pools[i][ptr:end_idx, :] = keys_list[i]
        else:
            remaining_space = self.size - ptr
            self.anchor_probs[ptr:self.size] = new_probs[:remaining_space]
            self.anchor_probs[0:end_idx - self.size] = new_probs[remaining_space:]
            
            for i in range(self.num_slice):
                self.anchor_pools[i][ptr:self.size, :] = keys_list[i][:remaining_space]
                self.anchor_pools[i][0:end_idx - self.size, :] = keys_list[i][remaining_space:]
        
        ptr = (ptr + batch_size) % self.size
        self.ptr[0] = ptr

    def forward(self, keys_batch_list: list[torch.Tensor], probs_batch: torch.Tensor = None):
        
        self._dequeue_and_enqueue(keys_batch_list, probs_batch)
        
        all_slice_keys = [pool.clone().detach() for pool in self.anchor_pools]
        slice_probs = self.anchor_probs.clone().detach()

        return all_slice_keys, slice_probs
    
    def get(self):
        
        all_slice_keys = [pool.clone().detach() for pool in self.anchor_pools]
        slice_probs = self.anchor_probs.clone().detach()

        return all_slice_keys, slice_probs


class SILPAG_model(nn.Module):
    def __init__(self, args):
        super(SILPAG_model, self).__init__()
        self.mae = nn.ModuleList([AE_encoder(hist=args.hist[i],
                                              img_size=args.img_size[i], 
                                              patch_size=args.patch_size[i], 
                                              dropout_rate=args.dropout_rate, 
                                              embed_dim=args.feat_dim, 
                                              decoder_embed_dim=args.feat_dim, 
                                              K=args.K[i]) for i in range(args.num_slice)])

        self.args = args
        

    def normalize_codebook(self):
        with torch.no_grad():
            for i in range(len(self.mae)):
                w = self.mae[i].codebook.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.mae[i].codebook.weight.copy_(w)

    def zero_code_usage(self):
        for mae in self.mae:
            mae.code_usage.zero_()

    def drop_useless_code(self):
        for i in range(len(self.mae)):
            print('='*20, 'Slice ', i, '='*20)
            print('    Original codebook size: ', self.mae[i].codebook.num_embeddings)

            usage = self.mae[i].code_usage
            non_zero_indices = torch.nonzero(usage, as_tuple=False).squeeze(1)
            newK = len(non_zero_indices)
            print('    New codebook size: ', newK)
            used_code = self.mae[i].codebook.weight[non_zero_indices]
            self.mae[i].codebook = nn.Embedding(newK, embedding_dim=self.mae[i].codebook.embedding_dim).to(self.args.device)
            with torch.no_grad():
                self.mae[i].codebook.weight.data.copy_(used_code)

            self.mae[i].code_usage = torch.zeros(newK, dtype=torch.float)

            self.args.K[i] = newK
        

    def Visualize_codeusage(self):
        for i in range(len(self.mae)):
            codebook = self.mae[i].get_codebook()
            self.visualize_codeusage(codebook, self.mae[i].code_usage, title='Codebook Usage in slice '+str(i), figsize=(5, 4))

    def visualize_codeusage(self, codebook, code_usage, title='Codebook usage', figsize=(5, 4)):
        """
        Visualize code usage
        """
        if isinstance(codebook, torch.Tensor):
            codebook = codebook.detach().cpu().numpy()
        if isinstance(code_usage, torch.Tensor):
            code_usage = code_usage.detach().cpu().numpy()
        
        reducer = PCA(n_components=2, random_state=42)
        
        codebook_2d = reducer.fit_transform(codebook)
        
        usage_norm = code_usage.astype(np.float32) / (code_usage.sum() + 1e-8)
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            x=codebook_2d[:, 0], 
            y=codebook_2d[:, 1], 
            c=usage_norm, 
            cmap='viridis', 
            alpha=0.7,
            s=25,
            vmin=0
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Code Usage', rotation=270, labelpad=15)
        
        plt.xlabel(f'PC 1')
        plt.ylabel(f'PC 2')
        plt.title(title, fontsize = 12)
        plt.grid(alpha=0.3)
        plt.show()

    def return_args(self):
        return self.args

    def forward(self, view, hist):
        # codebook = self.embedding_proj(self.codebook.weight)
        z = []
        vq = []
        ranking_loss, huber_loss, vq_loss = 0, 0, 0
        for i in range(len(self.mae)):
            z1, vq_weight, ranking_loss1, huber_loss1, vq_loss1 = self.mae[i](view[i], self.args.distr[i], hist[i])
            z.append(z1)
            vq.append(vq_weight)
            ranking_loss += ranking_loss1
            huber_loss += huber_loss1
            vq_loss += vq_loss1
            
        return z, vq, ranking_loss, huber_loss, vq_loss
    
    def forward_marker(self, view, hist):
        # codebook = self.embedding_proj(self.codebook.weight)
        z = []
        vq = []
        ranking_loss, huber_loss, vq_loss = 0, 0, 0
        for i in range(len(self.mae)):
            z1, vq_weight, ranking_loss1, huber_loss1, vq_loss1 = self.mae[i].forward_marker(view[i], self.args.distr[i], hist[i])
            z.append(z1)
            vq.append(vq_weight)
            ranking_loss += ranking_loss1
            huber_loss += huber_loss1
            vq_loss += vq_loss1
            
        return z, vq, ranking_loss, huber_loss, vq_loss

    def forward_wo_vq(self, view, hist):
        z = []
        ranking_loss, huber_loss, vq_loss = 0, 0, 0
        for i in range(len(self.mae)):
            z1, ranking_loss1, huber_loss1 = self.mae[i].forward_wo_vq(view[i], self.args.distr[i], hist[i])
            z.append(z1)
            ranking_loss += ranking_loss1
            huber_loss += huber_loss1
        return z, ranking_loss, huber_loss
    
    def forward_entmax(self, s, tgt_index):
        # s = -d
        return self.mae[tgt_index].forward_entmax(s)

    def get_genecode(self, s, tgt_index):
        sparse_s = self.forward_entmax(s, tgt_index)
        codebook = self.mae[tgt_index].get_codebook()

        genecode = sparse_s @ codebook
        return genecode
    
    def get_both(self, s, tgt_index): 
        sparse_s = self.forward_entmax(s, tgt_index)
        codebook = self.mae[tgt_index].get_codebook()
        genecode = sparse_s @ codebook
        return sparse_s, genecode

    def get_codebook(self, tgt_index):
        return self.mae[tgt_index].get_codebook()

    def generate(self, args, tgt: ad.AnnData, tgt_index, code_id='gene_code', gene='all'):
        adata = tgt.copy()
        if gene=='all':
            pass
        else:
            adata = adata[:, gene]
        z = adata.varm[code_id]
        test_loader = DataLoader(torch.tensor(z), batch_size=1, shuffle=False, drop_last=False)
        img = []
        # self.encoder1.code_usage.zero_()
        with torch.no_grad():
            for z in tqdm(test_loader, desc='Generating...', unit="gene"):
                z = z.to(args.device)
                pred, _, _, _ = self.mae[tgt_index].forward_decoder(z, args.distr[tgt_index])

                img.append(pred.squeeze(1).detach().cpu())
            
        img = torch.cat(img, dim=0)
        img = img.numpy()
        
        if args.resize_factor[tgt_index] > 1:
            resized_images = unpad_to_original(img, (img.shape[0], args.resized_img_size[tgt_index][1], args.resized_img_size[tgt_index][2]))
            img = np.empty((img.shape[0], args.orig_img_size[tgt_index][0], args.orig_img_size[tgt_index][1]), dtype=resized_images.dtype)
            for i in range(resized_images.shape[0]):
                full_image = np.kron(resized_images[i], np.ones((args.resize_factor[tgt_index], args.resize_factor[tgt_index])))
                img[i] = full_image[:args.orig_img_size[tgt_index][0], :args.orig_img_size[tgt_index][1]]
        else: 
            img = unpad_to_original(img, adata.varm['gene_img'].shape)
            
        X = img[:, adata.obs['array_y'].to_numpy(), adata.obs['array_x'].to_numpy()]
        X[X<=0] = 0
        adata.X = csr_matrix(X.T)
        return adata
    
    def generate1(self, args, tgt: ad.AnnData, tgt_index, embed_id='gene_embed', gene='all'):
        adata = tgt.copy()
        if gene=='all':
            pass
        else:
            adata = adata[:, gene]
        z = adata.varm[embed_id]
        test_loader = DataLoader(torch.tensor(z), batch_size=1, shuffle=False, drop_last=False)
        img = []
        # self.encoder1.code_usage.zero_()
        codebook = self.mae[tgt_index].get_codebook()
        for z in tqdm(test_loader, desc='Generating...', unit="gene"):
            z = z.to(args.device)
            z, _, _, _ = self.mae[tgt_index].forward_vq_sparse(z, codebook)
            pred, _, _, _ = self.mae[tgt_index].forward_decoder(z, args.distr[tgt_index])
            img.append(pred.squeeze(1))
            
        img = torch.cat(img, dim=0)
        img = img.data.cpu().numpy()
        
        if args.resize_factor[tgt_index] > 1:
            resized_images = unpad_to_original(img, (img.shape[0], args.resized_img_size[tgt_index][1], args.resized_img_size[tgt_index][2]))
            img = np.empty((img.shape[0], args.orig_img_size[tgt_index][0], args.orig_img_size[tgt_index][1]), dtype=resized_images.dtype)
            for i in range(resized_images.shape[0]):
                full_image = np.kron(resized_images[i], np.ones((args.resize_factor[tgt_index], args.resize_factor[tgt_index])))
                img[i] = full_image[:args.orig_img_size[tgt_index][0], :args.orig_img_size[tgt_index][1]]
        else: 
            img = unpad_to_original(img, adata.varm['gene_img'].shape)
            
        X = img[:, adata.obs['array_y'].to_numpy(), adata.obs['array_x'].to_numpy()]
        X[X<=0] = 0
        adata.X = csr_matrix(X.T)
        return adata
    
    def decode(self, args, tgt: ad.AnnData, code, tgt_index):
        adata = tgt.copy()
        adata = adata[:, :code.shape[0]]
        namelist = ['code'+str(i+1) for i in range(code.shape[0])]
        adata.var_names = namelist
        z = code
        test_loader = DataLoader(torch.tensor(z), batch_size=1, shuffle=False, drop_last=False)
        img = []
        # self.encoder1.code_usage.zero_()
        with torch.no_grad():
            for z in tqdm(test_loader, desc='Generating...', unit="gene"):
                z = z.to(args.device)
                pred, _, _, _ = self.mae[tgt_index].forward_decoder(z, args.distr[tgt_index])

                img.append(pred.squeeze(1).detach().cpu())
            
        img = torch.cat(img, dim=0)
        img = img.numpy()
        
        if args.resize_factor[tgt_index] > 1:
            resized_images = unpad_to_original(img, (img.shape[0], args.resized_img_size[tgt_index][1], args.resized_img_size[tgt_index][2]))
            img = np.empty((img.shape[0], args.orig_img_size[tgt_index][0], args.orig_img_size[tgt_index][1]), dtype=resized_images.dtype)
            for i in range(resized_images.shape[0]):
                full_image = np.kron(resized_images[i], np.ones((args.resize_factor[tgt_index], args.resize_factor[tgt_index])))
                img[i] = full_image[:args.orig_img_size[tgt_index][0], :args.orig_img_size[tgt_index][1]]
        else: 
            img = unpad_to_original(img, adata.varm['gene_img'].shape)
            
        X = img[:, adata.obs['array_y'].to_numpy(), adata.obs['array_x'].to_numpy()]
        X[X<=0] = 0
        adata.X = csr_matrix(X.T)
        return adata
    
    
class AE_encoder(nn.Module):
    """
    Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 hist = False, 
                 img_size=(28, 28),
                 patch_size=(4, 4),
                 dropout_rate=0.3,
                 embed_dim=64,
                 depth=3,
                 num_heads=4,
                 dim_head=4,
                 decoder_embed_dim=64,
                 mlp_ratio=4.,
                 K = 1024):
        super().__init__()

        self.hist = hist
        if self.hist:

            self.patch_embed_gene = PatchEmbed(img_size, patch_size, 1, embed_dim)
            self.patch_embed_hist = GeneWeightedPatchEmbed(img_size, patch_size, 1, 579, embed_dim)
            num_patches = self.patch_embed_gene.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

            self.blocks_gene = Block(embed_dim, num_heads, dim_head, mlp_ratio, qkv_bias=True)
            self.blocks_hist = Block(embed_dim, num_heads, dim_head, mlp_ratio, qkv_bias=True)
            self.fusion_blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, num_heads, dim_head, mlp_ratio, qkv_bias=True) for _ in range(depth-1)])
            
            self.output_proj = nn.Linear(embed_dim * 2, embed_dim)

        else:

            self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)  
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, dim_head, mlp_ratio, qkv_bias=True)
                for _ in range(depth)])

        self.codebook = nn.Embedding(K, embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=embed_dim**-0.5)
        self.embedding_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('code_usage', torch.zeros(K, dtype=torch.float))

        self.alpha_entmax = torch.tensor(1.6, requires_grad=False)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # decoder
        self.image_size = img_size
        self.patch_size = patch_size
        
        # self.decoder_rank = Decoder(embed_dim, self.image_size, dropout_rate)
        self.decoder_rank = DecoderCNN(embed_dim, self.image_size, dropout_rate)
        self.decoder_param = valueDecoder(int(embed_dim))

        self.pooling_weights = Parameter(torch.Tensor(((self.image_size[0]//self.patch_size[0])*(self.image_size[1]//self.patch_size[1])), 1))
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [(self.image_size[0] // self.patch_size[0]), (self.image_size[1] // self.patch_size[1])], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], [(self.image_size[0] // self.patch_size[0]), (self.image_size[1] // self.patch_size[1])], cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.hist:
            w = self.patch_embed_gene.proj.weight.data
        else:
            w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.constant_(self.pooling_weights, 1/((self.image_size[0]//self.patch_size[0])*(self.image_size[1]//self.patch_size[1])))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_codebook(self):
        codebook = self.embedding_proj(self.codebook.weight)
        codebook = F.normalize(codebook, p=2, dim=1)
        return codebook

    def forward_encoder(self, x, x_hist=None):
        """
        x_hist: (579, H, W)
        """
        if self.hist:
            x_hist = torch.unsqueeze(x_hist, dim=0)
            x_hist = self.patch_embed_hist(x, x_hist)
            x = self.patch_embed_gene(x)
            x = x + self.pos_embed[:, :, :]
            x_hist = x_hist + self.pos_embed[:, :, :]

            x = self.blocks_gene(x)
            x_hist = self.blocks_hist(x_hist)

            for blk in self.fusion_blocks:
                x, x_hist = blk(x, x_hist)

            # x = self.norm_gene(x)
            # x_hist = self.norm_hist(x_hist)
            x = torch.cat([x, x_hist], dim=-1) # (B, N, 2*D)
            x = self.output_proj(x)
        else: 
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, :, :]
            for blk in self.blocks:
                x = blk(x) 

        return x

    def forward_vq(self, x, codebook, soft=False):
        """
        x: tensor, [B, d]
        codebook: nn.Embedding, [K, d]
        """
        # x_ = self.pre_vq(x)
        d = torch.sum(x ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', x, rearrange(codebook, 'n d -> d n')) # [B, K]
        if soft:
            weights = F.softmax(-d, dim=1)
            self.code_usage += weights.sum(dim=0).to(self.code_usage)
            z_q = torch.mm(weights, codebook).view(x.shape)  # [B, d]
            vq_loss = 0.25 * torch.mean((z_q.detach() - x)**2) + torch.mean((z_q - x.detach())**2)
            z_q = x + (z_q - x).detach()
        else:
            min_encoding_indices = torch.argmin(d, dim=1)
            # codebook usage statistc
            counts = torch.bincount(min_encoding_indices, minlength=codebook.shape[0])
            self.code_usage += counts.to(self.code_usage)
            # z_q = codebook(min_encoding_indices).view(x.shape)
            z_q = F.embedding(min_encoding_indices, codebook).view(x.shape)
            vq_loss = torch.mean((z_q.detach()-x)**2) + 0.25 * torch.mean((z_q - x.detach()) ** 2)
            
            # preserve gradients
            z_q = x + (z_q - x).detach()
       #  z_q = self.post_vq(z_q)
        return z_q, vq_loss, d
    
    def forward_vq_sparse(self, x, codebook):
        """
        Args:
            x: [B, d]
            codebook: [K, d]
        
        Returns:
            z_q: [B, d]
            vq_loss
            weights: [B, K]
            d: [B, K]
        """
        # embed-code distance
        d = torch.sum(x ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', x, rearrange(codebook, 'n d -> d n')) # [B, K]
        s = -d   # similarity
        # s_min = torch.min(s, dim=1, keepdim=True)[0]
        # s_max = torch.max(s, dim=1, keepdim=True)[0]
        # s = (s - s_min) / (s_max - s_min + 1e-8)
        # s = torch.softmax(s, dim=1)
        # gamma = 1.0 
        # s = torch.exp(-gamma * d)
        sparse_s = self.forward_entmax(s)
        self.code_usage += torch.sum(sparse_s > 0, dim=0).to(self.code_usage)
        # sparse code
        z_q = sparse_s @ codebook  # [B, d]

        # commitment loss
        vq_loss = torch.mean((z_q.detach() - x)**2) + 0.25 * torch.mean((z_q - x.detach())**2)

        return z_q, vq_loss, s, d
    
    def forward_entmax(self, s):
        return entmax_bisect(s, alpha=self.alpha_entmax.to(s.device), dim=1, n_iter=24)

    def forward_decoder_rank(self, x):
        return self.decoder_rank(x)
    
    def forward_decoder_param(self, x):
        value1, value2 = self.decoder_param(x)
        return value1, value2
    
    def forward_decoder_gaussian(self, x):
        rank = self.forward_decoder_rank(x)
        log_mean, log_std = self.forward_decoder_param(x) # (N, 1)
        # mean = F.softplus(mean) + 1e-6
        # std = F.softplus(std) + 1e-6
        # std = torch.clamp(std, min=1e-6, max=1e3)

        mean = torch.exp(log_mean)
        std = torch.exp(log_std)
        x_rank = rank.view(rank.shape[0], -1) # [N, H*W]
        e = torch.randn_like(x_rank)
        x_ = mean + e * std # [N, H*W]
        x_ = torch.clamp(x_, min=0)

        sorted_indices = x_rank.argsort(dim=1).argsort(dim=1)
        x_ = torch.gather(torch.sort(x_, dim=1, descending=False).values, dim=1, index=sorted_indices)
        x_ = x_.view(rank.shape)
        return x_, rank, log_mean, log_std
    
    def forward_decoder_nb(self, x):
        rank = self.forward_decoder_rank(x)
        log_nb_mean, log_nb_dispersion = self.forward_decoder_param(x) # (N, 1)

        x_rank = rank.view(rank.shape[0], -1) # [N, L], L = H*W
        num_pixels = x_rank.shape[1]

        x_ = negbio_continuous_sampling(
            log_nb_mean, 
            log_nb_dispersion, 
            n_samples=num_pixels, 
            M=100,  
            tau=0.1
        )

        if torch.isnan(x_).any():
            print("NaN detected in decoder output! Clamping to zero.")
            x_ = torch.nan_to_num(x_, nan=0.0)

        sorted_indices = x_rank.argsort(dim=1).argsort(dim=1)
        x_ = torch.gather(torch.sort(x_, dim=1, descending=False).values, dim=1, index=sorted_indices)
        x_ = x_.view(rank.shape)
        return x_, rank, log_nb_mean, log_nb_dispersion
    
    def forward_decoder(self, x, distr):
        if distr == 'gaussian':
            return self.forward_decoder_gaussian(x)
        if distr == 'nb':
            return self.forward_decoder_nb(x)
        else:
            print('distr must be guassian or nb.')
    
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 1, H, W]
        pred: [N, L, H, W]
        """
        target = imgs.view(imgs.shape[0], -1)  # [N, H*W]
        pred = pred.view(imgs.shape[0], -1)    # [N, H*W]

        loss = F.huber_loss(pred, target, reduction='mean', delta=10.0)
        return loss
    
    def forward_param_loss(self, imgs, param1, param2, distr):
        """
        imgs: [N, 1, H, W]
        """
        target = imgs.view(imgs.shape[0], -1)  # [N, H*W]

        if distr == 'gaussian':
            mu = target.mean(dim=1, keepdim=True) 
            std = target.std(dim=1, unbiased=True, keepdim=True)
            mu = torch.clamp(mu, min=1e-6)
            std = torch.clamp(std, min=1e-6)
            
            log_mu = torch.log(mu)
            log_std = torch.log(std)
            loss = 0.5 * F.mse_loss(log_mu, param1) + 0.5 * F.mse_loss(log_std, param2)
            return loss
        
        if distr == 'nb':
            mu, dispersion = estimate_nb_params(target)
            mu = torch.clamp(mu, min=1e-6)
            log_mu = torch.log(mu)
            dispersion = torch.clamp(dispersion, min=1e-6, max=1e4)
            log_dispersion = torch.log(dispersion)
            loss = 0.5 * F.mse_loss(log_mu, param1) + 0.5 * F.mse_loss(log_dispersion, param2)
            return loss
        
        else:
            print('distr must be guassian or nb.')

    def forward_ranking_loss_pearson(self, imgs, pred):
        """
        imgs: [N, 1, H, W]
        pred: [N, 1, H, W]
        indicator: [N, H, W]
        """
        epsilon = 1e-8
        target = imgs.view(imgs.shape[0], -1) # [N, S]
        target_sorted, _ = torch.sort(target, dim=-1)
        target = torch.searchsorted(target_sorted, target, right=False).float()

        pred = pred.view(imgs.shape[0], -1)
        
        target = target - target.mean(dim=-1, keepdim=True)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        numerator = (target * pred).sum(dim=-1, keepdim=True)

        target_norm_sq = (target ** 2).sum(dim=-1, keepdim=True)
        pred_norm_sq = (pred ** 2).sum(dim=-1, keepdim=True)
        
        target_norm = torch.sqrt(target_norm_sq + epsilon)  # [N, L, 1]
        pred_norm = torch.sqrt(pred_norm_sq + epsilon)      # [N, L, 1]
        denominator = target_norm * pred_norm  # [N, L, 1]

        pearson = numerator / (denominator + epsilon)
        loss = 1 - pearson.mean()
        return loss
        
    def forward(self, imgs, distr, hist=None):
        """
        whole_latent: [N, L, feat_dim]
        """
        whole_latent = self.forward_encoder(imgs, hist)
        whole_latent = whole_latent + self.decoder_pos_embed

        sum_weight = torch.sum(self.pooling_weights)
        sum_weight = sum_weight + 1e-6 if sum_weight > 0 else sum_weight - 1e-6
        z = torch.sum(torch.mul(whole_latent, self.pooling_weights), dim=1) / sum_weight
        z = F.normalize(z, p=2, dim=1)
        codebook = self.get_codebook()
        z_q, vq_loss, vq_weight, d = self.forward_vq_sparse(z, codebook)
        rank = self.forward_decoder_rank(z_q)
        param1, param2 = self.forward_decoder_param(z_q)
        
        ranking_loss = self.forward_ranking_loss_pearson(imgs, rank)
        # huber_loss = self.forward_loss(imgs, pred)  
        huber_loss = self.forward_param_loss(imgs, param1, param2, distr)  
        return z, vq_weight, ranking_loss, huber_loss, vq_loss
    
    def forward_marker(self, imgs, distr, hist=None):
        """
        whole_latent: [N, L, feat_dim]
        latent: [N, L, feat_dim]
        """
        whole_latent = self.forward_encoder(imgs, hist)
        whole_latent = whole_latent + self.decoder_pos_embed
        sum_weight = torch.sum(self.pooling_weights)
        sum_weight = sum_weight + 1e-6 if sum_weight > 0 else sum_weight - 1e-6
        z = torch.sum(torch.mul(whole_latent, self.pooling_weights), dim=1) / sum_weight
        z = F.normalize(z, p=2, dim=1)

        codebook = self.get_codebook()
        z_q, vq_loss, d = self.forward_vq(z, codebook)
        rank = self.forward_decoder_rank(z_q)
        param1, param2 = self.forward_decoder_param(z_q)
        
        ranking_loss = self.forward_ranking_loss_pearson(imgs, rank)
        # huber_loss = self.forward_loss(imgs, pred)  
        huber_loss = self.forward_param_loss(imgs, param1, param2, distr)     
        return z, d, ranking_loss, huber_loss, vq_loss

    def forward_wo_vq(self, imgs, distr, hist=None):
        """
        whole_latent: [N, L, feat_dim]
        latent: [N, L, feat_dim]
        """
        whole_latent = self.forward_encoder(imgs, hist)
        whole_latent = whole_latent + self.decoder_pos_embed
        sum_weight = torch.sum(self.pooling_weights)
        sum_weight = sum_weight + 1e-6 if sum_weight > 0 else sum_weight - 1e-6
        z = torch.sum(torch.mul(whole_latent, self.pooling_weights), dim=1) / sum_weight
        z = F.normalize(z, p=2, dim=1)

        rank = self.forward_decoder_rank(z)
        param1, param2 = self.forward_decoder_param(z)
        
        ranking_loss = self.forward_ranking_loss_pearson(imgs, rank)
        # huber_loss = self.forward_loss(imgs, pred)  
        huber_loss = self.forward_param_loss(imgs, param1, param2, distr)    
        return z, ranking_loss, huber_loss


class valueDecoder(nn.Module):
    def __init__(self, feat_dim):
        super(valueDecoder, self).__init__()
        self.value1_layers = nn.Sequential(
            nn.Linear(feat_dim, int(feat_dim * 2)),
            # nn.LayerNorm(int(feat_dim * 2)),
            nn.ReLU(),
            nn.Linear(int(feat_dim * 2), 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Softplus()
        )
        self.value2_layers = nn.Sequential(
            nn.Linear(feat_dim, int(feat_dim * 2)),
            # nn.LayerNorm(int(feat_dim * 2)),
            nn.ReLU(),
            nn.Linear(int(feat_dim * 2), 64),
            # nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Softplus()
        )
    
    def forward(self, x):
        value1 = self.value1_layers(x)
        value2 = self.value2_layers(x)
        return value1, value2


class Decoder(nn.Module):
    def __init__(self, embed_dim, img_size, dropout_rate=0):
        super(Decoder, self).__init__()

        self.image_size = img_size  # 图像尺寸 (height, width)

        # 定义 decoder_Gene 网络部分
        self.decoder_Gene = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU())
        
        self.decoder_idf = nn.Linear(embed_dim, 512)

        self.decoder_sig = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, img_size[0] * img_size[1]),
            nn.ReLU()
        )

    def forward(self, x):
        gene_features = self.decoder_Gene(x)
        idf_features = self.decoder_idf(x)
        combined_features = gene_features + idf_features
        decoded = self.decoder_sig(combined_features)
        return decoded.view(-1, 1, self.image_size[0], self.image_size[1])


class DecoderCNN(nn.Module):
    def __init__(self, embed_dim, img_size, dropout_rate=0):
        super(DecoderCNN, self).__init__()
        self.image_size = img_size
        
        self.init_size = (img_size[0] // 4, img_size[1] // 4)
        self.fc = nn.Linear(embed_dim, 128 * self.init_size[0] * self.init_size[1])
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(size=img_size),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.size(0), 128, self.init_size[0], self.init_size[1])
        img = self.conv_blocks(out)
        return img


class PatchEmbed(nn.Module):  # [B, 3, 32, 32]->[B, 64, 256]
    def __init__(self, img_size=(32, 32), patch_size=(4, 4), in_chans=3, embed_dim=108):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.bn = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if torch.isnan(x).any():
            print('There is NaN in x!')
        x = self.proj(x)
        # x = self.bn(x)
        x = x.flatten(2).transpose(2, 1)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in 'patchembed'! Stopping training.")
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            nn.GELU()
        )

    def forward(self, x):
        output = self.net(x)
        return output


class MSA(nn.Module): # Multi-heads Self-Attention
    """
    dim is the input dimension, which is the width of embeding
    heads is how many patches there are
    dim_head is the number of dim required for each patch
    dropout is an argument to nn.Dropout()
    """

    def __init__(self, dim, heads=4, dim_head=2, dropout=0., attn_drop=0., qkv_bias=False):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.Dh = dim_head ** -0.5

        # the Wq, Wk, and Wv matrices in self-attention
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.linear_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.output = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        """
        param input: The input is embeding, [batch, N, D]
        return: The dimension of the MSA result is the same as the input dimension
        """

        # caculate q k v
        # [batch, N, inner_dim]
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)

        # switch to a multi-head attention mechanism
        new_shape = q.size()[:-1] + (self.heads, self.dim_head)
        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)
        q = torch.transpose(q, -3, -2)
        k = torch.transpose(k, -3, -2)
        v = torch.transpose(v, -3, -2)  # [batch, head, N, head_size]

        # caculate matrix A
        A = torch.matmul(q, torch.transpose(k, -2, -1)) * self.Dh
        A = torch.softmax(A, dim=-1)  # [batch,head, N, N]
        A = self.attn_drop(A)
        SA = torch.matmul(A, v)  # [batch,head, N, head_size]

        # multi-head attention mechanism concatenation
        SA = torch.transpose(SA, -3, -2)  # [batch, N,head, head_size]
        new_shape = SA.size()[:-2] + (self.inner_dim,)
        SA = SA.reshape(new_shape)  # [batch, N, inner_dim]
        out = self.output(SA)  # [batch, N, D]
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_ratio, qkv_bias=True):
        super(Block, self).__init__()
        hidden_dim = int(mlp_ratio * dim)
        self.norm = nn.LayerNorm(dim)
        self.msa = MSA(dim, heads=num_heads, dim_head=dim_head, qkv_bias=qkv_bias)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, input):
        output = self.norm(input)
        output = self.msa(output)
        output_s1 = output + input
        output = self.norm(output_s1)
        output = self.mlp(output)
        output_s2 = output + output_s1
        return output_s2
    

class MCA(nn.Module): # Multi-heads Cross-Attention
    """
    与 MSA 类似, 但 forward 接收两个输入:
    - x_q: 用于生成 Query 的序列
    - x_kv: 用于生成 Key 和 Value 的序列
    """
    def __init__(self, dim, heads=4, dim_head=2, dropout=0., attn_drop=0., qkv_bias=False):
        super(MCA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.Dh = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        
        # Q 来自 x_q, K/V 来自 x_kv
        self.linear_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.output = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv):
        """
        :param x_q: [batch, N_q, D] (Query 序列)
        :param x_kv: [batch, N_kv, D] (Key/Value 序列)
        :return: [batch, N_q, D]
        """
        
        # 计算 q, k, v
        q = self.linear_q(x_q) # [batch, N_q, inner_dim]
        k = self.linear_k(x_kv) # [batch, N_kv, inner_dim]
        v = self.linear_v(x_kv) # [batch, N_kv, inner_dim]

        # 切换到多头
        new_shape_q = q.size()[:-1] + (self.heads, self.dim_head)
        new_shape_kv = k.size()[:-1] + (self.heads, self.dim_head)
        
        q = q.view(new_shape_q)
        k = k.view(new_shape_kv)
        v = v.view(new_shape_kv)
        
        q = torch.transpose(q, -3, -2) # [batch, head, N_q, head_size]
        k = torch.transpose(k, -3, -2) # [batch, head, N_kv, head_size]
        v = torch.transpose(v, -3, -2) # [batch, head, N_kv, head_size]

        # 计算 A
        # (B, H, N_q, H_s) @ (B, H, H_s, N_kv) -> (B, H, N_q, N_kv)
        A = torch.matmul(q, torch.transpose(k, -2, -1)) * self.Dh
        A = torch.softmax(A, dim=-1)
        A = self.attn_drop(A)
        
        # (B, H, N_q, N_kv) @ (B, H, N_kv, H_s) -> (B, H, N_q, H_s)
        SA = torch.matmul(A, v)

        # 多头拼接
        SA = torch.transpose(SA, -3, -2) # [batch, N_q, head, head_size]
        new_shape = SA.size()[:-2] + (self.inner_dim,)
        SA = SA.reshape(new_shape) # [batch, N_q, inner_dim]
        out = self.output(SA) # [batch, N_q, D]
        return out
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_ratio, qkv_bias=True):
        super(CrossAttentionBlock, self).__init__()
        
        self.norm_g = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

        self.ca_g_queries_h = MCA(dim, num_heads, dim_head, qkv_bias=qkv_bias)
        
        self.ca_h_queries_g = MCA(dim, num_heads, dim_head, qkv_bias=qkv_bias)

        hidden_dim = int(mlp_ratio * dim)
        self.mlp_g = MLP(dim, hidden_dim)
        self.mlp_h = MLP(dim, hidden_dim)

    def forward(self, tokens_gene, tokens_hist):

        norm_tokens_gene = self.norm_g(tokens_gene)
        norm_tokens_histo = self.norm_h(tokens_hist)
        
        fused_gene = self.ca_g_queries_h(
            norm_tokens_gene, 
            norm_tokens_histo
        ) + tokens_gene 

        fused_hist = self.ca_h_queries_g(
            norm_tokens_histo, 
            norm_tokens_gene
        ) + tokens_hist
        
        out_gene = self.mlp_g(self.norm_g(fused_gene)) + fused_gene
        out_histo = self.mlp_h(self.norm_h(fused_hist)) + fused_hist

        return out_gene, out_histo
    

class GeneWeightedPatchEmbed(nn.Module):

    def __init__(self, img_size=(28, 28), patch_size=(4, 4), 
                 in_chans_gene=1, in_chans_hist=579, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans_hist = in_chans_hist
        self.embed_dim = embed_dim

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.K = self.patch_size[0] * self.patch_size[1] 

        # (B, 1, H, W) -> (B, 1*K, N)
        self.unfold_gene = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        # (1, 579, H, W) -> (1, 579*K, N)
        self.unfold_hist = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.proj = nn.Linear(in_chans_hist, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, gene_img, hist_feat):
        
        C_hist = hist_feat.shape[1]

        # (B, 1, H, W) -> (B, K, N)
        gene_patches_unfold = self.unfold_gene(gene_img)
        # (B, K, N) -> (B, N, K)
        gene_patches = gene_patches_unfold.transpose(1, 2)
        
        # (B, N, K) -> (B, N, K)
        weights = F.softmax(gene_patches, dim=-1)

        # (1, C, H, W) -> (1, C*K, N)
        hist_patches_unfold = self.unfold_hist(hist_feat)
        # (1, C*K, N) -> (1, C, K, N)
        hist_patches = hist_patches_unfold.reshape(1, C_hist, self.K, self.num_patches)

        # (1, C, K, N) -> (1, N, C, K)
        # (Batch, Patches, Features, Pixels)
        hist_patches = hist_patches.permute(0, 3, 1, 2)

        weighted_hist_tokens = torch.einsum('bnk, bnck -> bnc', weights, hist_patches)
        
        x = self.proj(weighted_hist_tokens)
        x = self.norm(x)
        return x


# get positional embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: list of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def negbio_continuous_sampling(log_mean, log_dispersion, n_samples, M=100, tau=0.1):
    """
    input_size: [Batch, 1]
    out_put_size: [Batch, n_samples]
    """
    mean = torch.exp(log_mean)
    dispersion = torch.exp(log_dispersion)
    # mean = torch.clamp(mean, min=1e-5, max=1e5) 
    # dispersion = torch.clamp(dispersion, min=1e-5, max=1e5)

    eps = 1e-6
    rate = dispersion / (mean + eps)

    # Lambda ~ Gamma(r, beta)
    # dispersion: [N, 1] -> [N, n_samples]
    # rate: [N, 1] -> [N, n_samples]
    r_expanded = dispersion.expand(-1, n_samples)
    rate_expanded = rate.expand(-1, n_samples)

    if torch.isnan(r_expanded).any() or torch.isnan(rate_expanded).any():
        r_expanded = torch.nan_to_num(r_expanded, nan=0.5)
        rate_expanded = torch.nan_to_num(rate_expanded, nan=0.5)

    r_expanded = torch.clamp(r_expanded, min=1e-5, max=1e8)
    rate_expanded = torch.clamp(rate_expanded, min=1e-5, max=1e8)

    gamma_dist = Gamma(concentration=r_expanded, rate=rate_expanded)
    lambdas = gamma_dist.rsample() # [N, n_samples]
    lambdas = torch.clamp(lambdas, min=1e-5, max=1e5)

    # Poisson process
    exp_dist = Exponential(rate=lambdas)
    
    # [M, N, n_samples]
    inter_arrival_times = exp_dist.rsample((M,))
    
    cumulative_times = torch.cumsum(inter_arrival_times, dim=0) 
    
    # sum(sigmoid((1 - S_n) / tau))
    soft_counts = torch.sigmoid((1.0 - cumulative_times) / tau)
    z_tilde = torch.sum(soft_counts, dim=0) # [N, n_samples]
    
    return z_tilde


def negbio_continuous_sampling_rp(r, p, n_samples, M=100, tau=0.1):
    """
    r (dispersion), p (probability) of Negative Binomial 
    """

    if torch.isnan(r).any() or torch.isnan(p).any():
        r = torch.nan_to_num(r, nan=1.0)
        p = torch.nan_to_num(p, nan=0.5)

    r = torch.clamp(r, min=1e-5, max=1e5)
    
    p = torch.clamp(p, min=1e-2, max=1.0 - 1e-2)

    epsilon = 1e-8
    gamma_rate = p / torch.clamp(1.0 - p, min=epsilon)

    r_expanded = r.expand(-1, n_samples)
    rate_expanded = gamma_rate.expand(-1, n_samples)
    rate_expanded = torch.clamp(rate_expanded, min=1e-8, max=1e8)

    gamma_dist = Gamma(concentration=r_expanded, rate=rate_expanded)
    lambdas = gamma_dist.rsample() # [N, n_samples]

    if torch.isnan(lambdas).any():
        print("Warning: NaN in lambdas, replacing with 1.0")
        lambdas = torch.nan_to_num(lambdas, nan=1.0)
    
    lambdas = torch.clamp(lambdas, min=1e-5, max=1e5)

    exp_dist = Exponential(rate=lambdas)
    
    inter_arrival_times = exp_dist.rsample((M,))
    
    cumulative_times = torch.cumsum(inter_arrival_times, dim=0) 
    
    soft_counts = torch.sigmoid((1.0 - cumulative_times) / tau)
    z_tilde = torch.sum(soft_counts, dim=0) # [N, n_samples]
    
    return z_tilde


def estimate_nb_params(data: torch.Tensor, eps: float = 1e-6, max_dispersion: float = 1e4):
    if not data.is_floating_point():
        data = data.float()

    mu = data.mean(dim=1, keepdim=True) 
    var = data.var(dim=1, unbiased=True, keepdim=True)

    var_minus_mu = var - mu

    var_minus_mu = torch.clamp(var_minus_mu, min=eps)

    mu_sq = mu.square()
    dispersion = mu_sq / var_minus_mu

    dispersion = torch.clamp(dispersion, min=1e-4, max=max_dispersion)
    
    return mu, dispersion

