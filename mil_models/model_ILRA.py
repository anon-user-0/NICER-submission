import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import numpy as np

from .components import process_clf, process_surv
"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""

def save2pkl(data, pkl_path):
    folder = os.path.dirname(pkl_path)
    os.makedirs(folder, exist_ok=True)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K, attn_mask=None):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, attn = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O, attn


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X, attn_mask=None):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H, _ = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat, _ = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret, attn = self.mha(global_embedding, X)
        return ret, attn


class ILRA(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        
        self.config = config
        self.mode = mode
        
        self.n_classes = self.config.n_classes
        
        
        # default settings
        self.num_layers = 2
        self.hidden_feat=64
        self.num_heads=8
        self.topk=2
        self.ln=False
        
        self.dim_in = 1024 # hardcode for testing
        # self.dim_in = config.in_dim
        
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(self.num_layers):
            block = GAB(dim_in=self.dim_in if idx == 0 else self.hidden_feat,
                        dim_out=self.hidden_feat,
                        num_heads=self.num_heads,
                        num_inds=self.topk,
                        ln=self.ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=self.hidden_feat, num_heads=self.num_heads, ln=self.ln)

        # classifier
        self.classifier = nn.Linear(in_features=self.hidden_feat, out_features=self.n_classes)
        
    def get_top_attn(self, attn_weight):
        # Step 1: Reshape to 1D array of size M
        weights = attn_weight.cpu().detach().numpy().flatten()  # or attention_weights[0, 0, :]
        M = weights.shape[0]
        # Step 2: Get indices sorted by weights in descending order
        sorted_indices = np.argsort(weights)[::-1]  # [::-1] reverses for descending order

        # Step 3: Select top 50% indices
        top_50_percent = int(np.ceil(M / 2))  # Number of indices for top 50%
        top_indices = sorted_indices[:top_50_percent]
        
        return top_indices
        

    def forward_no_loss(self, x, attn_mask=None):
        if len(x.shape)==2: # DeepSets
            x = x.unsqueeze(1)
        if attn_mask is not None:
            x = x[:, :int(torch.sum(attn_mask[:, 0, :]).item()), :]
        if x.shape[-1] > self.dim_in:
            x = x[:, :, 1:self.dim_in+1]
            
        
        for block in self.gab_blocks:
            x = block(x)

        feat, attn = self.pooling(x)
        
        logits = self.classifier(feat)

        logits = logits.squeeze(1)
        out = {'logits': logits}
        return out
    
    def forward(self, h, model_kwargs={}):
        h = h.float()
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']
            results_dict, log_dict = process_clf(logits, label, loss_fn)
            
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
            
        else:
            raise NotImplementedError("Not Implemented!")

        return results_dict, log_dict
    
    def forward_ig(self, h):
        out = self.forward_no_loss(h)
        logits = out['logits']
        logits = torch.softmax(logits, dim=-1)
        return logits