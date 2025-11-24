import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .model_configs import ABMILConfig


class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.in_dim = 1024
        
        self.embed_dim=config.embed_dim
        self.mlp = create_mlp(in_dim=self.in_dim,
                              hid_dims=[config.embed_dim] *
                              (config.n_fc_layers - 1),
                              dropout=config.dropout,
                              out_dim=config.embed_dim,
                              end_with_fc=False)
        
        print(config.n_fc_layers)

        if config.gate:
            self.attention_net = Attn_Net_Gated(L=self.embed_dim,
                                                D=config.attn_dim,
                                                dropout=config.dropout,
                                                n_classes=1)
        else:
            self.attention_net = Attn_Net(L=self.embed_dim,
                                          D=config.attn_dim,
                                          dropout=config.dropout,
                                          n_classes=1)

        self.classifier = nn.Linear(self.embed_dim, config.n_classes)
        self.n_classes = config.n_classes

        self.mode = mode

    def forward_attention(self, h, attn_only=False):
        # B: batch size
        # N: number of instances per WSI
        # L: input dimension
        # K: number of attention heads (K = 1 for ABMIL)
        # h is B x N x L
        h = h.float()
        h = self.mlp(h)
        # h is B x N x D
        A = self.attention_net(h)  # B x N x K
        A = torch.transpose(A, -2, -1)  # B x K x N 
        if attn_only:
            return A
        else:
            return h, A

    def forward_no_loss(self, h, attn_mask=None):
        # if len(h.shape)==2:
        #     h = h.unsqueeze(-1).repeat(1, 1, self.in_dim)
        if len(h.shape)==2:
            h = h.unsqueeze(1)
            # h = h.reshape(1, 16, -1)
        if h.shape[-1] > self.in_dim:
            h = h[:, :, 1:self.in_dim+1]

        h, A = self.forward_attention(h)
        A_raw = A
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.bmm(A, h).squeeze(dim=1) # B x K x C --> B x C
        logits = self.classifier(M)

        out = {'logits': logits, 'attn': A, 'feats': h, 'feats_agg': M}
        return out
    
    def forward(self, h, model_kwargs={}):

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
        return logits
