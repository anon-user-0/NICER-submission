"""
Hard-clustering-based aggregation

Ref:
    Vu, Quoc Dang, et al. "Handcrafted Histological Transformer (H2T): Unsupervised representation of whole slide images." Medical image analysis 85 (2023): 102743.
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pdb

from tqdm import tqdm
from .components import predict_clf, predict_surv, predict_emb
from utils.file_utils import save_pkl, load_pkl

class DeepSets(nn.Module):
    """
    WSI is represented as a mean of patch vector set
    """
    def __init__(self, config, mode):
        super().__init__()

        assert config.load_proto, "Prototypes must be loaded!"
        assert os.path.exists(config.proto_path), "Path {} doesn't exist!".format(config.proto_path)

        self.config = config
        self.mode = mode
        proto_path = config.proto_path

        if proto_path.endswith('pkl'):
            weights = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            weights = np.load(proto_path)

        self.n_proto = config.out_size
        self.prototypes = torch.from_numpy(weights).float()
        self.prototypes = self.prototypes / torch.norm(self.prototypes, dim=1).unsqueeze(1)

        emb_dim = config.in_dim

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out = x.mean(dim=1) # BxNxD -> BxD
        return {'repr': out}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']

    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y