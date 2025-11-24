"""
Hard-clustering-based aggregation

Ref:
    Vu, Quoc Dang, et al. "Handcrafted Histological Transformer (H2T): Unsupervised representation of whole slide images." Medical image analysis 85 (2023): 102743.
"""

import torch
import torch.nn as nn
import os
import pdb

from sklearn.datasets import load_iris
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from .components import predict_clf, predict_clf_nonparam, predict_surv, predict_emb, predict_surv_nonparam
from utils.file_utils import save_pkl, load_pkl

class dpmeans:
    def __init__(self, X, init_proto, max_iter=50):
        """
        Simple DP-means implementation with an upper bound on #clusters.

        Args:
            X           : (N, D) data
            init_proto  : max number of clusters (K_init)
        """
        n, d = X.shape
        self.d = d

        # upper bound for number of clusters
        self.K_init = int(init_proto)
        assert self.K_init >= 1

        # start with a single cluster
        self.K = 1
        self.z = np.zeros(n, dtype=int)  # labels in [0, K-1]

        # cluster centers, start with mean of the data
        self.mu = np.array([np.mean(X, axis=0)])  # shape (1, d)

        self.sigma = 1.0
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K) / self.K

        # penalty lambda via k-means++-style init
        # self.Lambda = self.kpp_init(X, self.K_init)
        self.Lambda = 1000

        self.max_iter = max_iter
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)

    def kpp_init(self, X, k):
        """
        k-means++ style initialization to estimate lambda:
        lambda = max distance to any of the k++ means.
        """
        n, d = X.shape
        mu = np.zeros((k, d))
        dist = np.inf * np.ones(n)

        # choose the first center uniformly at random
        idx0 = np.random.randint(0, n)
        mu[0, :] = X[idx0, :]

        for i in range(1, k):
            D = X - mu[i - 1, :][None, :]
            dist = np.minimum(dist, np.sum(D * D, axis=1))
            probs = dist / float(np.sum(dist))
            cum_probs = np.cumsum(probs)
            r = np.random.rand()
            idx = np.searchsorted(cum_probs, r)
            mu[i, :] = X[idx, :]

        Lambda = float(np.max(dist))
        return Lambda

    def fit(self, X):
        """
        Run DP-means on X.

        Returns:
            z       : (N,) cluster labels in [0, K-1]
            mu      : (K, D) cluster centers
            em_time : (max_iter,) timing info (mostly zeros here)
        """

        print("Lambda:", self.Lambda)
        print("max distance to mean:", np.max(np.sum((X - np.mean(X,0))**2, axis=1)))
        n, d = X.shape
        obj_tol = 1e-3
        max_iter = self.max_iter

        obj = np.zeros(max_iter)
        # em_time = np.zeros(max_iter)

        for it in range(max_iter):
            # tic = time.time()

            # compute distances to each cluster center
            dist = np.zeros((n, self.K))
            for kk in range(self.K):
                Xm = X - self.mu[kk, :][None, :]
                dist[:, kk] = np.sum(Xm * Xm, axis=1)

            # assignment step
            dmin = np.min(dist, axis=1)
            self.z = np.argmin(dist, axis=1)

            # create a new cluster if far from all existing ones
            idx = np.where(dmin > self.Lambda)[0]
            if idx.size > 0 and self.K < self.K_init:
                self.K += 1
                new_center = np.mean(X[idx, :], axis=0, keepdims=True)  # (1, d)
                self.mu = np.vstack([self.mu, new_center])              # (K, d)

                # recompute distances including the new center
                Xm = X - self.mu[self.K - 1, :][None, :]
                new_dist = np.sum(Xm * Xm, axis=1)[:, None]  # (n, 1)
                dist = np.hstack([dist, new_dist])

                # re-assign those far points to the new cluster
                self.z[idx] = self.K - 1

            # update step: recompute centers and mixing proportions
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                idxk = np.where(self.z == kk)[0]
                self.nk[kk] = len(idxk)
                if len(idxk) > 0:
                    self.mu[kk, :] = np.mean(X[idxk, :], axis=0)

            self.pik = self.nk / float(np.sum(self.nk))

            # compute objective
            for kk in range(self.K):
                idxk = np.where(self.z == kk)[0]
                obj[it] += np.sum(dist[idxk, kk])
            obj[it] += self.Lambda * self.K

            # em_time[it] = time.time() - tic

            # convergence check
            if it > 0 and np.abs(obj[it] - obj[it - 1]) < obj_tol * obj[it]:
                break

        self.obj = obj
        # self.em_time = em_time
        return self.z, self.mu, 0

    def compute_nmi(self, z1, z2):
        """
        Normalized mutual information between two labelings.
        """
        z1 = np.asarray(z1)
        z2 = np.asarray(z2)
        n = z1.size

        labels1 = np.unique(z1)
        labels2 = np.unique(z2)
        k1 = labels1.size
        k2 = labels2.size

        nk1 = np.zeros((k1, 1))
        nk2 = np.zeros((k2, 1))

        for i, lab in enumerate(labels1):
            nk1[i] = np.sum(z1 == lab)
        for j, lab in enumerate(labels2):
            nk2[j] = np.sum(z2 == lab)

        pk1 = nk1 / float(np.sum(nk1))
        pk2 = nk2 / float(np.sum(nk2))

        nk12 = np.zeros((k1, k2))
        for i, lab1 in enumerate(labels1):
            for j, lab2 in enumerate(labels2):
                nk12[i, j] = np.sum((z1 == lab1) & (z2 == lab2))
        pk12 = nk12 / float(n)

        eps = np.finfo(float).eps
        Hx = -np.sum(pk1 * np.log(pk1 + eps))
        Hy = -np.sum(pk2 * np.log(pk2 + eps))
        Hxy = -np.sum(pk12 * np.log(pk12 + eps))

        MI = Hx + Hy - Hxy
        nmi = MI / (0.5 * (Hx + Hy))
        return nmi


class DPMeans(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, config, mode):
        super().__init__()

        assert config.load_proto, "Prototypes must be loaded!"
        assert os.path.exists(config.proto_path), "Path {} doesn't exist!".format(config.proto_path)

        self.config = config
        self.mode = mode
        # proto_path = config.proto_path

        # if proto_path.endswith('pkl'):
        #     weights = load_pkl(proto_path)['prototypes'].squeeze()
        # elif proto_path.endswith('npy'):
        #     weights = np.load(proto_path)

        self.n_proto = config.out_size
        # self.prototypes = torch.from_numpy(weights).float()
        # self.prototypes = self.prototypes / torch.norm(self.prototypes, dim=1).unsqueeze(1)

        emb_dim = config.in_dim

    def dprocess(self, x):
        self.dp = dpmeans(x.cpu().numpy(), self.n_proto)
        z, mu, em_time = self.dp.fit(x.cpu().numpy())   
        print(mu.shape)  

        return mu, z

    def representation(self, x, idx=None):
        """
        Construct unsupervised slide representation
        """
        feats, dist = self.dprocess(x.squeeze(0))
        out = torch.tensor(feats)
        dist = torch.tensor(dist)
        out = out.unsqueeze(0)  # (1, K, D)
        return {'repr': out, 'qq': dist}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']

    def predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, y, _, qqs = predict_clf_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y, _, qqs = predict_surv_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, y, qqs