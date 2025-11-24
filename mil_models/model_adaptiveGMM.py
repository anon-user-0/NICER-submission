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
from .components import predict_clf, predict_clf_nonparam, predict_surv_nonparam, predict_emb
from utils.file_utils import save_pkl, load_pkl
from sklearn.mixture import BayesianGaussianMixture

def adaptive_gmm(
    X,
    max_components=20,
    weight_threshold=1e-2,
    random_state=0
):
    """
    Adaptive GMM using a truncated Dirichlet Process approximation.

    Returns
    -------
    bgmm : fitted BayesianGaussianMixture model
    labels : (N,) hard cluster assignments
    n_effective_clusters : int
    effective_means : (K_eff, D)
    effective_stds  : (K_eff, D)  # diagonal std per cluster
    """
    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0,
        init_params="kmeans",
        max_iter=50,
        random_state=random_state
    )
    bgmm.fit(X)

    # Identify used components
    used = bgmm.weights_ > weight_threshold
    n_effective_clusters = int(np.sum(used))

    # Means
    effective_means = bgmm.means_[used]            # (K_eff, D)

    # Standard deviations = sqrt(diagonal of cov matrices)
    effective_stds = bgmm.covariances_[used]       # (K_eff, D, D)

    effective_features = np.concatenate([effective_means, effective_stds], axis=1)  # (K_eff, 2D)

    return effective_features, bgmm.predict(X)


class AdaptiveGMM(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, config, mode):
        super().__init__()

        self.mode = mode
        self.n_proto = config.out_size
        emb_dim = config.in_dim


    def representation(self, x, idx=None):
        """
        Construct unsupervised slide representation
        """
        feats, dist = adaptive_gmm(x.squeeze(0).cpu().numpy(), 
                                   max_components=32,
                                    weight_threshold=1e-2)
        out = torch.tensor(feats)   # K_eff, 2D
        dist = torch.tensor(dist)   
        out = out.unsqueeze(0)  # (1, K_eff, 2D)
        return {'repr': out, 'qq': dist}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']

    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y, _, qq = predict_clf_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y, qq = predict_surv_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y, qq