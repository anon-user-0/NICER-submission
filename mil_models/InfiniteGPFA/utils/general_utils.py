from typing import Optional

import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .kernels import Kernel
from .constants import (
    float_type, 
    jitter_level
)


def probit(x: torch.Tensor):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2e-3) + 1e-3


def gaussian_KL_wrt_standard_normal(q_mu: torch.Tensor, q_lmd_sqrt: torch.Tensor):
    # analytically compute KL divergence between a Gaussian distribution a standard normal distribution
    L = torch.tril(torch.permute(q_lmd_sqrt, (2, 0, 1)))
    kl = 0.5 * (
        torch.sum(torch.square(q_mu)) - 
        torch.prod(torch.tensor(q_lmd_sqrt.shape[1:])) - 
        torch.sum(torch.log(torch.square(torch.diagonal(L, dim1=-2, dim2=-1)))) + 
        torch.sum(torch.square(L))
    )

    return kl


def diagonal_gaussian_KL_wrt_standard_normal(q_mu: torch.Tensor, q_lmd_sqrt: torch.Tensor):
    # analytically compute KL divergence between a diagonal Gaussian distribution a standard normal distribution
    kl = 0.5 * (
        torch.sum(torch.square(q_mu)) - 
        torch.prod(torch.tensor(q_lmd_sqrt.shape)) - 
        torch.sum(torch.log(torch.square(q_lmd_sqrt))) +
        torch.sum(torch.square(q_lmd_sqrt))
    )
    
    return kl


def gaussian_KL(q_mu: torch.Tensor, q_lmd_sqrt: torch.Tensor, K: torch.Tensor):
    L = torch.linalg.cholesky(K, upper=False)
    alpha = torch.linalg.solve_triangular(L, q_mu, upper=False)
    num_trials = q_lmd_sqrt.shape[2]
    Lq = torch.tril(torch.permute(q_lmd_sqrt, (2, 0, 1)))
    L_tiled = torch.tile(L[None], (Lq.shape[0], 1, 1))
    L_inverse_Lq = torch.linalg.solve_triangular(L_tiled, Lq, upper=False)
    
    kl = 0.5 * (
        torch.sum(torch.square(alpha)) + # quadrative term
        num_trials * torch.sum(torch.log(torch.square(torch.diagonal(L, dim1=-2, dim2=-1)))) - #log determinant prior
        torch.prod(torch.tensor(q_lmd_sqrt.shape[1:])) - # constant term
        torch.sum(torch.log(torch.square(torch.diagonal(Lq, dim1=-2, dim2=-1)))) + # log determinant variational
        torch.sum(torch.square(L_inverse_Lq)) # trace term
    )
    
    return kl


def diagonal_gaussian_KL(q_mu: torch.Tensor, q_lmd_sqrt: torch.Tensor, K: torch.Tensor):
    L = torch.cholesky(K, upper=False)
    alpha = torch.linalg.solve_triangular(L, q_mu, upper=False)
    num_trials = q_lmd_sqrt.shape[1]
    L_inverse = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper=False)
    K_inverse = torch.linalg.solve_triangular(L.T, L_inverse, upper=True)
    
    kl = 0.5 * (
        torch.sum(torch.square(alpha)) + # quadratic term
        num_trials * torch.sum(torch.log(torch.square(torch.diagonal(L, dim1=-2, dim2=-1)))) - # log determinant prior
        torch.prod(torch.tensor(q_lmd_sqrt.shape)) - # constant term
        torch.sum(torch.log(torch.square(q_lmd_sqrt))) + # log determinant variational
        torch.sum(torch.diagonal(K_inverse)[None] * torch.square(q_lmd_sqrt)) # trace term
    )
    
    return kl


def gp_conditional(
    X_new: torch.Tensor, 
    X: torch.Tensor, 
    kernel: Kernel, 
    f: torch.Tensor, 
    full_cov: bool = False, 
    q_lmd_sqrt: Optional[torch.Tensor] = None, 
    full_whitening: bool = False, 
):
    N = X.shape[0]
    # X_new = X_new / X_new.max()
    Kmn = kernel.K(X, X_new)
    Kmm = kernel.K(X) + torch.eye(N, dtype=float_type, device=X.device) * jitter_level
    Lm = torch.linalg.cholesky(Kmm, upper=False)
    
    A = torch.linalg.solve_triangular(Lm, Kmn, upper=False)
    
    if full_cov:
        f_var = kernel.K(X_new) - torch.matmul(A.T, A)
        target_shape = (f.shape[1], 1, 1)
    else:
        f_var = kernel.K_diagonal(X_new) - torch.sum(torch.square(A), dim=0)
        target_shape = (f.shape[1], 1)
    f_var = torch.tile(f_var[None], target_shape)
    
    # A_temp = torch.linalg.solve_triangular(torch.transpose(Lm, 0, 1), A, upper=True)
    # f_mean = torch.matmul(A_temp.T, f)
    # f_mean = torch.matmul(A.T, f)
    if full_whitening:
        A_temp = torch.linalg.solve_triangular(torch.transpose(Lm, 0, 1), A, upper=True)
        f_mean = torch.matmul(A_temp.T, f)
    else:
        A = torch.linalg.solve_triangular(torch.transpose(Lm, 0, 1), A, upper=True)
        f_mean = torch.matmul(A.T, f)
    
    if q_lmd_sqrt is not None:
        if len(q_lmd_sqrt.shape) == 2:
            LT_A = A * q_lmd_sqrt.T[..., None]
        elif len(q_lmd_sqrt.shape) == 3:
            L = torch.tril(torch.permute(q_lmd_sqrt, (2, 0, 1)))
            A_tiled = torch.tile(A[None], (f.shape[1], 1, 1))
            LT_A = torch.matmul(torch.transpose(L, dim0=-2, dim1=-1), A_tiled)
        else:
            raise ValueError
        
        if full_cov:
            f_var = f_var + torch.matmul(torch.transpose(LT_A, dim0=-2, dim1=-1), LT_A)
        else:
            f_var = f_var + torch.sum(torch.square(LT_A), dim=1)
    
    f_var = torch.transpose(f_var, dim0=-2, dim1=-1)
    
    return f_mean, f_var


def gp_conditional_v2(
    X_new: torch.Tensor, 
    X: torch.Tensor, 
    kernel: Kernel, 
    f: torch.Tensor, 
    full_cov: bool = False, 
    q_lmd_sqrt: Optional[torch.Tensor] = None, 
    whitening: bool = False, 
):
    N = X.shape[0]
    Kmn = kernel.K(X, X_new)
    Kmm = kernel.K(X) #  + torch.eye(N, dtype=float_type) * jitter_level
    Kmm_inv = torch.linalg.inv(Kmm)
    q_lmd_sqrt = torch.permute(q_lmd_sqrt, (2, 0, 1))
    S = torch.matmul(q_lmd_sqrt, torch.transpose(q_lmd_sqrt, -2, -1))
    
    f_mean = Kmn.T @ Kmm_inv @ f
    f_cov = kernel.K(X_new) - Kmn.T @ Kmm_inv @ Kmn + Kmn.T @ Kmm_inv @ S @ Kmm_inv @ Kmn
    
    # f_mean = torch.matmul(torch.matmul(Kmn.T, Kmm_inv), f)
    # f_cov = kernel.K(X_new) - torch.matmul(Kmn.T, torch.matmul(Kmm_inv, Kmn)) + torch.matmul(Kmn.T, torch.matmul(Kmm_inv, torch.matmul(S, torch.matmul(Kmm_inv, Kmn))))
    f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1)
    
    f_var = f_var.T
    
    return f_mean, f_var
    
    
def compare_responsibilities_to_mask(responsibilities: np.ndarray, true_mask: np.ndarray):
    # Ensure inputs are numpy arrays
    resp = np.array(responsibilities)
    mask = np.array(true_mask)
    
    # Calculate similarity matrix using log-likelihood
    # We keep this as a 2D matrix
    similarity_matrix = np.zeros((resp.shape[1], mask.shape[1]))
    for i in range(resp.shape[1]):
        for j in range(mask.shape[1]):
            similarity_matrix[i, j] = np.sum(
                mask[:, j] * np.log(resp[:, i] + 1e-10) + 
                (1 - mask[:, j]) * np.log(1 - resp[:, i] + 1e-10)
            )
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
    
    # Reorder the responsibilities
    reordered_resp = resp[:, col_ind]
    
    # Calculate agreement metrics
    auc_scores = [roc_auc_score(mask[:, i], reordered_resp[:, i]) for i in range(mask.shape[1])]
    ap_scores = [average_precision_score(mask[:, i], reordered_resp[:, i]) for i in range(mask.shape[1])]
    
    return {
        'permutation': col_ind,
        'auc_scores': auc_scores,
        'average_auc': np.mean(auc_scores),
        'ap_scores': ap_scores,
        'average_ap': np.mean(ap_scores)
    }


def CCA_analysis(latents: np.ndarray, behavioural_data: np.ndarray, n_components: int):
    scaler_latents = StandardScaler()
    scaler_behavioural = StandardScaler()
    
    standardised_latents = scaler_latents.fit_transform(latents)
    standardised_behavioural = scaler_behavioural.fit_transform(behavioural_data)
    
    cca = CCA(n_components=n_components)
    cca.fit(standardised_latents, standardised_behavioural)
    
    latents_cca, behaviour_cca = cca.transform(standardised_latents, standardised_behavioural)
    
    canonical_correlations = cca.score(standardised_latents, standardised_behavioural)
    
    latents_weights = cca.x_weights_
    behavioural_weights = cca.y_weights_
    
    return (
        cca, 
        standardised_latents, 
        standardised_behavioural, 
        latents_cca, 
        behaviour_cca, 
        canonical_correlations, 
        latents_weights,
        behavioural_weights, 
    )
