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

class KMeans(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.n_proto = config.out_size

        emb_dim = config.in_dim

    def local_kmeans(self, s, mode='faiss', n_iter=30, n_init=5, sampling=1000):
        assert sampling > 0, "cannot perform 0-means clustering"
        if mode=='kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=sampling, max_iter=n_iter, n_init=n_init)
            kmeans.fit(s.cpu())
            weight = kmeans.cluster_centers_
            labels = kmeans.labels_
        elif mode == 'faiss':
            try:
                import faiss
            except ImportError:
                print("FAISS not installed.")
                raise
                
            numOfGPUs = torch.cuda.device_count()
            kmeans = faiss.Kmeans(s.shape[1],
                                  sampling,
                                  niter=n_iter,
                                  nredo=5,
                                  verbose=False,
                                  gpu=numOfGPUs)
            kmeans.train(s.cpu().numpy())
            weight = kmeans.centroids
            labels = 0   # faiss is not used so it's placeholder
        else:
            raise NotImplementedError(f"Clustering not implemented for {mode}!")

        return weight, labels

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        feats, dist = self.local_kmeans(x.squeeze(0), mode='kmeans',
                                         n_iter=30, n_init=5, sampling=self.n_proto)
        out = torch.tensor(feats)
        dist = torch.tensor(dist)
        out = out.reshape(x.shape[0], -1)
        return {'repr': out, 'qq': dist}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']

    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y, qq = predict_clf(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y, qq = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y, qq