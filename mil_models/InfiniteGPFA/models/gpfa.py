from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from ..utils.likelihoods import Likelihood
from ..utils.kernels import Kernel, SquaredExponential
from ..utils.mean_functions import ZeroMeanFunction
from ..utils.constants import int_type, float_type, std_qmu_init, jitter_level
from ..utils.general_utils import (
    gaussian_KL_wrt_standard_normal, 
    diagonal_gaussian_KL_wrt_standard_normal, 
    gaussian_KL, 
    diagonal_gaussian_KL, 
    gp_conditional, 
)


class VariationalGPFA(nn.Module):
    def __init__(
        self, 
        X: np.ndarray, # inputs
        Y: np.ndarray, # observations
        kernels: List[Kernel], # kernels
        likelihood: Likelihood, # conditional likelihood
        C: torch.Tensor, 
        d: torch.Tensor, 
        units_out: Optional[torch.Tensor] = None, 
        m_step: bool = True, 
    ):
        super().__init__()
        self.kernels = kernels
        self.likelihood = likelihood
        
        self.D = len(kernels)
        
        self.mean_functions = [ZeroMeanFunction() for _ in range(self.D)]
        self.f_inds = [0 for _ in range(self.D)]
        
        self.X = X
        self.Y = Y
        
        self.N, self.obs_dim, self.num_trials = Y.shape
        
        self.units_out = torch.tensor([], dtype=torch.int64) if units_out is None else torch.as_tensor(units_out, dtype=torch.int64)
        mask = torch.ones((self.N, self.obs_dim, self.num_trials), dtype=float_type)
        mask[:, self.units_out, :] = 0
        self.mask = mask
        
        self.initialise_parameters(C, d, m_step)
        self.initialise_inference()
    
    def initialise_parameters(self, C: torch.Tensor, d: torch.Tensor, m_step: bool=True):
        self.C = nn.Parameter(C, requires_grad=m_step)
        self.d = nn.Parameter(d, requires_grad=m_step)
    
    def initialise_inference(self):
        """
        The variational distribution is parametrised following the brilliant trick noticed in 
        Opper & Archambeau (2009).
        I.e., q(f) = N(f|K\alpha + m(\cdot), [K^{-1} + \Lambda]^{-1})
        where \Lambda is a diagonal matrix
        """
        self.q_alpha = nn.Parameter(torch.zeros((self.D, self.N, self.num_trials), dtype=float_type), requires_grad=True)
        self.q_lambda = nn.Parameter(torch.ones((self.D, self.N, self.num_trials), dtype=float_type), requires_grad=True)
        
    def variational_free_energy(self):
        # firstly compute the KL-divergence between the variational approximation and the prior
        prior_kl = self.compute_prior_kl()
        _, _, log_rates_mean, log_rates_var = self.predict_log_rates(self.X)
        exp_ll = self.likelihood.variational_expectations(log_rates_mean, log_rates_var, self.Y)
        free_energy = torch.sum(exp_ll * self.mask) - prior_kl
        
        return -free_energy
        
    def predict_log_rates(self, X_new: Optional[torch.Tensor]=None, full_cov: bool=False):
        f_means, f_covs = [], []
        for d in range(self.D):
            Kx = self.kernels[d].K(self.X, X_new)
            K = self.kernels[d].K(self.X)
            
            # predictive f_mean
            f_mean = torch.matmul(torch.transpose(Kx, -2, -1), self.q_alpha[d]) + self.mean_functions[d](X_new)
            
            # predictive f_cov
            A = K + torch.diag_embed(1.0 / torch.square(self.q_lambda[d].T))
            # A = K + torch.stack([torch.eye(self.N) / torch.square(self.q_lambda[d, :, r]) for r in range(self.num_trials)])
            L = torch.linalg.cholesky(A, upper=False) + jitter_level * torch.eye(self.N, dtype=float_type)
            Kx = torch.tile(Kx[None], (self.num_trials, 1, 1))
            L_inverse_Kx = torch.linalg.solve_triangular(L, Kx, upper=False)
            
            if full_cov:
                f_cov = self.kernels[d].K(X_new) - torch.matmul(L_inverse_Kx.transpose(), L_inverse_Kx)
            else:
                # f_cov = self.kernels[d].K(X_new) - torch.sum(torch.square(L_inverse_Kx), dim=1)
                f_cov = self.kernels[d].K_diagonal(X_new) - torch.sum(torch.square(L_inverse_Kx), dim=1)
            
            f_means.append(f_mean)
            f_covs.append(torch.transpose(f_cov, -2, -1))
        
        f_means = torch.stack(f_means)
        f_covs = torch.stack(f_covs)
        
        log_rates_mean = torch.einsum("md,dnr->nmr", self.C, f_means) + self.d[None, :]
        log_rates_var = torch.einsum("md,dnr->nmr", torch.square(self.C), f_covs)
        
        return f_means, f_covs, log_rates_mean, log_rates_var
    
    def compute_prior_kl(self):
        prior_kl = 0.0
        
        for d in range(self.D):
            # mean of the variational Gaussian approximation
            K = self.kernels[d].K(self.X)
            K_alpha = torch.matmul(K, self.q_alpha[d, :, :])
            f_mean = K_alpha + self.mean_functions[d](self.X)
            
            # covariance matrix of the variational Gaussian approximation
            I = torch.tile(torch.eye(self.N)[None], (self.num_trials, 1, 1)) * (1 + jitter_level)
            A = I + torch.transpose(self.q_lambda[d, :, :], 0, 1)[:, None] *\
                torch.transpose(self.q_lambda[d, :, :], 0, 1)[..., None] * K
            L = torch.linalg.cholesky(A, upper=False)
            L_inverse = torch.linalg.solve_triangular(L, I, upper=False)
            temp = L_inverse / torch.transpose(self.q_lambda[d], 0, 1)[:, None]
            f_cov = 1. / torch.square(self.q_lambda[d]) - torch.transpose(torch.sum(torch.square(temp), dim=1), 0, 1)
            
            A_logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)))
            trace_A_inverse = torch.sum(torch.square(L_inverse))
            
            prior_kl += 0.5 * (A_logdet + trace_A_inverse - self.N * self.num_trials + torch.sum(K_alpha * self.q_alpha[d]))
        
        return prior_kl


class SparseVariationalGPFA(nn.Module):
    def __init__(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        kernels: List[Kernel], 
        likelihood: Likelihood, 
        Z: List[torch.Tensor], 
        C: torch.Tensor, 
        d: torch.Tensor, 
        whitening: bool = True, 
        q_diagonal: bool = True, 
        units_out: Optional[torch.Tensor] = None, 
        m_step: bool = True, 
        train_inducing_locs: bool = True, 
        sigmasq: float = 1.0, 
    ):
        super().__init__()
        
        self.X = X
        self.Y = Y
        
        if train_inducing_locs:
            self.Z = nn.ParameterList(Z)
        else:
            self.Z = Z
        
        self.kernels = nn.ModuleList(kernels)
        self.likelihood = likelihood
        
        self.D = len(kernels)
        self.mean_functions = [ZeroMeanFunction() for _ in range(self.D)]
        
        self.num_inducing = [z.shape[0] for z in Z]
        
        self.whitening = whitening
        self.q_diagonal = q_diagonal
        self.f_indices = [0 for _ in range(self.D)] # shared covariate
        
        self.N, self.M, self.num_trials = Y.shape
        
        self.units_out = torch.tensor([], dtype=torch.int64) if units_out is None else units_out
        mask = torch.ones((self.N, self.M, self.num_trials), dtype=float_type)
        mask[:, self.units_out, :] = 0
        self.mask = mask
        
        self.sigmasq = sigmasq
        
        self.initialise_inference()
        self.initialise_parameters(C, d, m_step)
        
    def initialise_parameters(self, C: torch.Tensor, d: torch.Tensor, m_step: bool=True):
        self.C = nn.Parameter(C, requires_grad=m_step)
        self.d = nn.Parameter(d, requires_grad=m_step)
    
    def initialise_inference(self):
        self.q_mu = nn.ParameterList([])
        self.q_lmd_sqrt = nn.ParameterList([])
        
        for d in range(self.D):
            self.q_mu.append(
                nn.Parameter(
                    torch.randn(self.num_inducing[d], self.num_trials, dtype=float_type) * std_qmu_init, 
                    requires_grad=True, 
                )
            )

            if self.q_diagonal:
                q_lmd_sqrt = torch.ones((self.num_inducing[d], self.num_trials), dtype=float_type)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
            else:
                q_lmd_sqrt = torch.swapaxes(torch.stack([torch.eye(self.num_inducing[d], dtype=float_type) for _ in range(self.num_trials)]), 0, 2)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
    
    def variational_free_energy(self):
        prior_kl = self.compute_prior_KL()
        _, _, log_rates_mean, log_rates_var = self.predict_log_rates(self.X)
        exp_ll = self.likelihood.variational_expectations(log_rates_mean, log_rates_var, self.Y)
        
        free_energy = torch.sum(exp_ll * self.mask) - prior_kl
        
        return -free_energy
        
    def compute_prior_KL(self):
        prior_kl = 0.0
        for d in range(self.D):
            if self.whitening:
                if self.q_diagonal:
                    prior_kl = prior_kl + diagonal_gaussian_KL_wrt_standard_normal(self.q_mu[d], self.q_lmd_sqrt[d])
                else:
                    prior_kl = prior_kl + gaussian_KL_wrt_standard_normal(self.q_mu[d], self.q_lmd_sqrt[d])
            else:
                K = self.kernels[d].K(self.Z[self.f_indices[d]]) + torch.eye(self.num_inducing[d]) * jitter_level
                if self.q_diagonal:
                    prior_kl = prior_kl + diagonal_gaussian_KL(self.q_mu[d], self.q_lmd_sqrt[d], K)
                else:
                    prior_kl = prior_kl + gaussian_KL(self.q_mu[d], self.q_lmd_sqrt[d], K)
        
        return prior_kl
    
    def predict_f(self, X_new: torch.Tensor, full_cov: bool = False):
        mus, vars = [], []
        for d in range(self.D):
            x = X_new[:, self.f_indices[d]].reshape(-1, 1)
            mu, var = gp_conditional(
                X_new=x, 
                X=self.Z[self.f_indices[d]], 
                kernel=self.kernels[d], 
                f=self.q_mu[d], 
                full_cov=full_cov, 
                q_lmd_sqrt=self.q_lmd_sqrt[d], 
                full_whitening=True, 
            )
            mus.append(mu + self.mean_functions[d](x))
            vars.append(var)
        
        return torch.stack(mus), torch.stack(vars)
    
    def predict_log_rates(self, X_new: torch.Tensor):
        f_means, f_vars = self.predict_f(X_new, full_cov=False)
        
        log_rates_mean = torch.einsum("md,dnr->nmr", self.C, f_means) + self.d[None]
        log_rates_var = torch.einsum("md,dnr->nmr", torch.square(self.C), f_vars)
        
        return f_means, f_vars, log_rates_mean, log_rates_var


class InfiniteSparseVariationalGPFA(nn.Module):
    def __init__(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        D: int,
        likelihood: Likelihood, 
        Z: List[torch.Tensor], 
        C: torch.Tensor, 
        d: torch.Tensor, 
        alpha: float = 2.0, 
        q_diagonal: bool = True, 
        units_out: Optional[torch.Tensor] = None, 
        m_step: bool = True, 
        train_inducing_locs: bool = True, 
        train_alpha: bool = False, 
        C_prior: Optional[Dict[str, Any]] = None, 
        d_prior: Optional[Dict[str, Any]] = None, 
        alpha_prior: Optional[Dict[str, Any]] = None, 
        lmd_orthogonality: float = 0.0, 
    ):
        super().__init__()
        
        self.X = X
        self.Y = Y
        
        if train_inducing_locs:
            self.Z = nn.ParameterList(Z)
        else:
            self.Z = Z
            
        self.kernels = nn.ModuleList()
        for d_ in range(D):
            kernel = SquaredExponential(input_dim=1, variance=1.0, lengthscales=0.005)
            kernel.to(X.device)
            self.kernels.append(kernel)
            
        self.likelihood = likelihood
        
        self.D = len(self.kernels)
        self.mean_functions = [ZeroMeanFunction() for _ in range(self.D)]
        
        self.num_inducing = [z.shape[0] for z in Z]
        
        self.q_diagonal = q_diagonal # we assume gp prior on inducing points regardless (i.e., no whitening)
        self.f_indices = [0 for _ in range(self.D)]
        
        self.N, self.M, self.num_trials = Y.shape
        
        self.units_out = torch.tensor([], dtype=torch.int64) if units_out is None else units_out
        mask = torch.ones((self.N, self.M, self.num_trials), dtype=float_type)
        mask[:, self.units_out, :] = 0
        self.mask = mask
        
        self.C_prior = C_prior
        self.d_prior = d_prior
        self.alpha_prior = alpha_prior
        
        self.lmd_orthogonality = lmd_orthogonality
        
        self.initialise_inference()
        self.initialise_parameters(C, d, alpha, m_step, train_alpha)
        
    def initialise_parameters(
        self, 
        C: torch.Tensor, 
        d: torch.Tensor, 
        alpha: float, 
        m_step: bool = True, 
        train_alpha: bool = False, 
    ):
        if self.C_prior is None:
            self.C = nn.Parameter(C, requires_grad=m_step)
        if self.d_prior is None:
            self.d = nn.Parameter(d, requires_grad=m_step)
        if self.alpha_prior is None:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=float_type), requires_grad=train_alpha)
        
    def initialise_inference(self):
        self.q_mu = nn.ParameterList([])
        self.q_lmd_sqrt = nn.ParameterList([])
        
        for d in range(self.D):
            self.q_mu.append(
                nn.Parameter(
                    torch.randn(self.num_inducing[d], self.num_trials, dtype=float_type) * std_qmu_init, 
                    requires_grad=True, 
                )
            )

            if self.q_diagonal:
                q_lmd_sqrt = torch.ones((self.num_inducing[d], self.num_trials), dtype=float_type)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
            else:
                q_lmd_sqrt = torch.swapaxes(torch.stack([torch.eye(self.num_inducing[d], dtype=float_type) for _ in range(self.num_trials)]), 0, 2)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
        
        self.q_a = nn.Parameter(torch.ones((self.D, ), dtype=float_type), requires_grad=True)
        self.q_b = nn.Parameter(torch.ones((self.D, ), dtype=float_type), requires_grad=True)
        
        self.q_tau_logit = nn.Parameter(torch.zeros((self.N, self.D, self.num_trials), dtype=float_type), requires_grad=True)
        
        if self.C_prior is not None:
            assert "prior_variance" in self.C_prior
            
            self.C_prior_mu = torch.zeros((self.M, self.D), dtype=float_type)
            C_var = self.C_prior["prior_variance"]
            self.C_prior_var = torch.ones((self.D, ), dtype=float_type) * C_var
            
            self.C_q_mu = torch.nn.Parameter(torch.randn(self.M, self.D, dtype=float_type) * 0.01, requires_grad=True)
            self.C_q_lmd_sqrt = torch.nn.Parameter(torch.ones((self.M, self.D), dtype=float_type), requires_grad=True)
            
        if self.d_prior is not None:
            assert "prior_variance" in self.d_prior
            
            self.d_prior_mu = torch.zeros((self.M, ), dtype=float_type)
            d_var = self.d_prior["prior_variance"]
            self.d_prior_var = torch.ones((self.M, ), dtype=float_type) * d_var
            
            self.d_q_mu = torch.nn.Parameter(torch.randn(self.M, dtype=float_type) * 0.01, requires_grad=True)
            self.d_q_lmd_sqrt = torch.nn.Parameter(torch.ones((self.M, ), dtype=float_type), requires_grad=True)
        
        if self.alpha_prior is not None:
            assert "s1" in self.alpha_prior and "s2" in self.alpha_prior
            
            self.alpha_prior_s1 = torch.tensor(self.alpha_prior["s1"], dtype=float_type)
            self.alpha_prior_s2 = torch.tensor(self.alpha_prior["s2"], dtype=float_type)
            
            self.alpha_q_s1_exponent = torch.nn.Parameter(torch.tensor(0.0, dtype=float_type), requires_grad=True)
            self.alpha_q_s2_exponent = torch.nn.Parameter(torch.tensor(0.0, dtype=float_type), requires_grad=True)
        
    def variational_free_energy(self):
        kl = self.compute_KL()
        exp_ll = self.exp_conditional_ll(self.X)
        # exp_ll = self.exp_conditional_ll_v2(self.X)
        
        free_energy = exp_ll - kl
        
        loss = -free_energy
        
        if self.lmd_orthogonality > 0.0:
            loss = loss + self.lmd_orthogonality * self.orthogonality_penalty()
        
        return loss
    
    def compute_KL(self):
        kl_u = self.compute_KL_inducing()
        kl_pi = self.compute_KL_pi()
        exp_kl_Z = self.compute_expected_KL_Z()
        
        kl = kl_u + kl_pi + exp_kl_Z
        
        if self.C_prior is not None:
            kl_C = self.compute_KL_C()
            kl = kl + kl_C
        
        if self.d_prior is not None:
            kl_d = self.compute_KL_d()
            kl = kl + kl_d
        
        if self.alpha_prior is not None:
            kl_alpha = self.compute_KL_alpha()
            kl = kl + kl_alpha
        
        return kl
        
    def compute_KL_inducing(self):
        kl = 0.0
        for d in range(self.D):
            
            K = self.kernels[d].K(self.Z[self.f_indices[d]])
            K = K + torch.eye(self.num_inducing[d], device=K.device, dtype=float_type) * jitter_level
            
            if self.q_diagonal:
                kl = kl + diagonal_gaussian_KL(self.q_mu[d], self.q_lmd_sqrt[d], K)
            else:
                kl = kl + gaussian_KL(self.q_mu[d], self.q_lmd_sqrt[d], K)
        
        return kl
    
    def compute_KL_pi(self):
        kl = 0.0
        if self.alpha_prior is not None:
            alpha_q_s1 = torch.exp(self.alpha_q_s1_exponent)
            alpha_q_s2 = torch.exp(self.alpha_q_s2_exponent)
            alpha = alpha_q_s1 / alpha_q_s2
            log_alpha = torch.digamma(alpha_q_s1) - self.alpha_q_s2_exponent
        else:
            alpha = self.alpha
            log_alpha = torch.log(self.alpha)
        for d in range(self.D):
            kl = kl + torch.lgamma(self.q_a[d] + self.q_b[d]) - torch.lgamma(self.q_a[d]) - torch.lgamma(self.q_b[d]) - log_alpha + np.log(self.D) + \
                (self.q_a[d] - alpha / self.D) * (torch.digamma(self.q_a[d]) - torch.digamma(self.q_a[d] + self.q_b[d])) + \
                    (self.q_b[d] - 1) * (torch.digamma(self.q_b[d]) - torch.digamma(self.q_a[d] + self.q_b[d]))
        
        return kl
    
    def compute_KL_C(self):
        kl = 0.0
        for d in range(self.D):
            kl = kl + 0.5 * (
                self.M * torch.log(self.C_prior_var[d]) - 
                self.M - 
                torch.sum(torch.log(torch.square(self.C_q_lmd_sqrt[:, d]))) + 
                1 / self.C_prior_var[d] * (torch.sum(torch.square(self.C_q_lmd_sqrt[:, d])) + torch.sum(torch.square(self.C_q_mu[:, d])))
            )
        return kl
    
    def compute_KL_d(self):
        kl = 0.5 * torch.sum(
            torch.square(self.d_q_lmd_sqrt) / self.d_prior_var - 
            1.0 + 
            torch.square(self.d_q_mu) / self.d_prior_var + 
            torch.log(self.d_prior_var) - 
            torch.log(torch.square(self.d_q_lmd_sqrt))
        )
        
        return kl
    
    def compute_KL_alpha(self):
        alpha_q_s1 = torch.exp(self.alpha_q_s1_exponent)
        alpha_q_s2 = torch.exp(self.alpha_q_s2_exponent)
        
        kl = (alpha_q_s1 - self.alpha_prior_s1) * torch.digamma(alpha_q_s1) - torch.lgamma(alpha_q_s1) + torch.lgamma(self.alpha_prior_s1) + \
            self.alpha_prior_s1 * (torch.log(alpha_q_s2) - torch.log(self.alpha_prior_s2)) + alpha_q_s1 * (self.alpha_prior_s2 / alpha_q_s2 - 1)
            
        return kl
    
    def compute_expected_KL_Z(self):
        q_tau = torch.sigmoid(self.q_tau_logit)
        kl = torch.sum(q_tau * torch.log(q_tau) + (1 - q_tau) * torch.log(1 - q_tau) - 
                       q_tau * (torch.digamma(self.q_a[None, :, None]) - torch.digamma(self.q_a[None, :, None] + self.q_b[None, :, None])) - 
                       (1 - q_tau) * (torch.digamma(self.q_b[None, :, None]) - torch.digamma(self.q_a[None, :, None] + self.q_b[None, :, None]))
                       )
        
        return kl
    
    def orthogonality_penalty(self):
        q_tau = torch.sigmoid(self.q_tau_logit)
        return torch.sum(torch.abs(q_tau))
    
    def predict_f(self, X_new: torch.Tensor, full_cov: bool = False):
        mus, vars = [], []
        for d in range(self.D):
            x = X_new[:, self.f_indices[d]].reshape(-1, 1)
            mu, var = gp_conditional(
                X_new=x, 
                X=self.Z[self.f_indices[d]], 
                kernel=self.kernels[d], 
                f=self.q_mu[d], 
                full_cov=full_cov, 
                q_lmd_sqrt=self.q_lmd_sqrt[d], 
                full_whitening=False, 
            )
            
            mus.append(mu + self.mean_functions[d](x))
            # print(mus[-1].max(), mus[-1].min())
            vars.append(var)
        
        return torch.stack(mus), torch.stack(vars) # (D, N, R)
    
    def predict_log_rates(self, X_new: torch.Tensor):
        f_means, f_vars = self.predict_f(X_new, full_cov=False)
        # f_means = f_means - torch.mean(f_means, dim=1, keepdims=True) / (torch.std(f_means, dim=1, keepdims=True) + 1e-6)
        
        q_tau = torch.sigmoid(self.q_tau_logit)
        if self.C_prior is not None:
            C = self.C_q_mu
        else:
            C = self.C
        if self.d_prior is not None:
            d = self.d_q_mu
        else:
            d = self.d
        # print(f_means.max(), f_means.min())
        log_rates_mean = torch.einsum("md,ndr,dnr->nmr", C, q_tau, f_means) + d[None]
        
        q_tau = torch.transpose(q_tau, dim0=0, dim1=1)
        f_z_var = torch.square(q_tau) * f_vars + (torch.square(f_means)) * q_tau * (1 - q_tau)
        if self.C_prior is not None:
            log_rates_var = torch.einsum(
                "md,dnr,dnr->nmr", 
                torch.square(self.C_q_lmd_sqrt), 
                q_tau, 
                torch.square(f_means) + f_vars
            ) + \
                torch.einsum(
                    "md,dnr->nmr", 
                    torch.square(self.C_q_mu), 
                    q_tau * torch.square(f_means) + q_tau * (1 - q_tau) * f_vars
                )
        else:
            C_squared = torch.square(self.C)
            log_rates_var = torch.einsum("md,dnr->nmr", C_squared, f_z_var)
        if self.d_prior is not None:
            log_rates_var = log_rates_var + torch.square(self.d_q_lmd_sqrt)[None, :, None]
        
        return f_means, f_vars, log_rates_mean, log_rates_var
    
    def exp_conditional_ll(self, X_new: torch.Tensor):
        f_means, f_vars, log_rates_mean, log_rates_var = self.predict_log_rates(X_new)
        
        exp_ll = self.likelihood.variational_expectations(log_rates_mean, log_rates_var, self.Y)
        

        self.mask = self.mask.to(exp_ll.device)
        exp_ll = torch.sum(exp_ll * self.mask)
        
        return exp_ll
    
    def exp_conditional_ll_v2(self, X_new: torch.Tensor):
        f_means, f_vars = self.predict_f(X_new, full_cov=False)  # f_means: [D, N, R], f_vars: [D, N, R]
        
        q_tau = torch.sigmoid(self.q_tau_logit)  # Shape: [N, D, R]
        
        # Term 1: y_n^T y_n
        term1 = torch.sum(self.Y**2)
        
        # Term 2: -2y_n^T (sum_d tau_nd mu_nd^f C_d)
        term2 = -2 * torch.sum(self.Y * torch.einsum('ndr,md,dnr->nmr', q_tau, self.C, f_means))
        
        # Term 3: sum_d tau_nd ((S_d^f)_nn + (mu_nd^f)^2) C_d^T C_d
        C_squared_sum = torch.sum(self.C**2, dim=0)  # Shape: [D]
        term3 = torch.sum(q_tau * (f_vars + f_means**2).permute(1, 0, 2) * C_squared_sum.unsqueeze(0).unsqueeze(-1))
        
        # Term 4: Cross terms
        C_product = torch.matmul(self.C.T, self.C)  # Shape: [D, D]
        mask = torch.tril(torch.ones_like(C_product), diagonal=-1)
        q_tau_product = q_tau.unsqueeze(2) * q_tau.unsqueeze(1)  # Shape: [N, D, D, R]
        f_means_product = f_means.permute(1, 0, 2).unsqueeze(2) * f_means.permute(1, 0, 2).unsqueeze(1)  # Shape: [N, D, D, R]
        term4 = 2 * torch.sum(q_tau_product * f_means_product * C_product.unsqueeze(0).unsqueeze(-1) * mask.unsqueeze(0).unsqueeze(-1))
        
        exp_ll = -(term1 + term2 + term3 + term4) / (2 * self.likelihood.variance)
        
        return exp_ll
    
    
class DoublyInfiniteSparseVariationalGPFA(InfiniteSparseVariationalGPFA):
    def __init__(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        kernels: List[Kernel], 
        likelihood: Likelihood, 
        Z: List[torch.Tensor], 
        C: torch.Tensor, 
        d: torch.Tensor, 
        alpha: float = 2.0, 
        q_diagonal: bool = True, 
        units_out: Optional[torch.Tensor] = None, 
        m_step: bool = True, 
        train_inducing_locs: bool = True, 
        train_alpha: bool = False, 
        C_prior: Optional[Dict[str, Any]] = None, 
        alpha_prior: Optional[Dict[str, Any]] = None, 
    ):
        super().__init__(
            X=X, 
            Y=Y, 
            kernels=kernels, 
            likelihood=likelihood, 
            Z=Z, 
            C=C, 
            d=d, 
            alpha=alpha, 
            q_diagonal=q_diagonal, 
            units_out=units_out, 
            m_step=m_step, 
            train_inducing_locs=train_inducing_locs, 
            train_alpha=train_alpha, 
            C_prior=C_prior, 
            alpha_prior=alpha_prior
        )
        
    def initialise_inference(self):
        self.q_mu = nn.ParameterList([])
        self.q_lmd_sqrt = nn.ParameterList([])
        
        for d in range(self.D):
            self.q_mu.append(
                nn.Parameter(
                    torch.randn(self.num_inducing[d], self.num_trials, dtype=float_type) * std_qmu_init, 
                    requires_grad=True, 
                )
            )

            if self.q_diagonal:
                q_lmd_sqrt = torch.ones((self.num_inducing[d], self.num_trials), dtype=float_type)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
            else:
                q_lmd_sqrt = torch.swapaxes(torch.stack([torch.eye(self.num_inducing[d], dtype=float_type) for _ in range(self.num_trials)]), 0, 2)
                self.q_lmd_sqrt.append(nn.Parameter(q_lmd_sqrt, requires_grad=True))
        
        self.q_a_raw = nn.Parameter(torch.zeros((self.D, ), dtype=float_type), requires_grad=True)
        self.q_b_raw = nn.Parameter(torch.zeros((self.D, ), dtype=float_type), requires_grad=True)
        
        self.q_tau_logit = nn.Parameter(torch.zeros((self.N, self.D, self.num_trials), dtype=float_type), requires_grad=True)
        
        if self.C_prior is not None:
            assert "prior_variance" in self.C_prior
            
            self.C_prior_mu = torch.zeros((self.M, self.D), dtype=float_type)
            C_var = self.C_prior["prior_variance"]
            self.C_prior_var = torch.ones((self.D, ), dtype=float_type) * C_var
            
            self.C_q_mu = torch.nn.Parameter(torch.randn(self.M, self.D, dtype=float_type) * 0.01, requires_grad=True)
            self.C_q_lmd_sqrt = torch.nn.Parameter(torch.ones((self.M, self.D), dtype=float_type), requires_grad=True)
            
        if self.d_prior is not None:
            assert "prior_variance" in self.d_prior
            
            self.d_prior_mu = torch.zeros((self.M, ), dtype=float_type)
            d_var = self.d_prior["prior_variance"]
            self.d_prior_var = torch.ones((self.M, ), dtype=float_type) * d_var
            
            self.d_q_mu = torch.nn.Parameter(torch.randn(self.M, dtype=float_type) * 0.01, requires_grad=True)
            self.d_q_lmd_sqrt = torch.nn.Parameter(torch.ones((self.M, ), dtype=float_type), requires_grad=True)
        
        if self.alpha_prior is not None:
            assert "s1" in self.alpha_prior and "s2" in self.alpha_prior
            
            self.alpha_prior_s1 = torch.tensor(self.alpha_prior["s1"], dtype=float_type)
            self.alpha_prior_s2 = torch.tensor(self.alpha_prior["s2"], dtype=float_type)
            
            self.alpha_q_s1_exponent = torch.nn.Parameter(torch.tensor(0.0, dtype=float_type), requires_grad=True)
            self.alpha_q_s2_exponent = torch.nn.Parameter(torch.tensor(0.0, dtype=float_type), requires_grad=True)
        
    def compute_KL(self):
        kl_u = self.compute_KL_inducing()
        kl_v = self.compute_KL_v()
        exp_kl_Z = self.compute_expected_KL_Z()
        
        kl = kl_u + kl_v + exp_kl_Z
        
        if self.C_prior is not None:
            kl_C = self.compute_KL_C()
            kl = kl + kl_C
        
        if self.alpha_prior is not None:
            kl_alpha = self.compute_KL_alpha()
            kl = kl + kl_alpha
        
        return kl
    
    def compute_KL_v(self):
        kl = 0.0
        # note that now q_a and q_b are the shape parameters for the Beta variational approximations over v_d instead of pi_d
        if self.alpha_prior is not None:
            alpha_q_s1 = torch.exp(self.alpha_q_s1_exponent)
            alpha_q_s2 = torch.exp(self.alpha_q_s2_exponent)
            alpha = alpha_q_s1 / alpha_q_s2
            log_alpha = torch.digamma(alpha_q_s1) - self.alpha_q_s2_exponent
        else:
            alpha = self.alpha
            log_alpha = torch.log(self.alpha)
        q_a = torch.exp(self.q_a_raw)
        q_b = torch.exp(self.q_b_raw)
        for d in range(self.D):
            kl = kl + torch.lgamma(q_a[d] + q_b[d]) - torch.lgamma(q_a[d]) - torch.lgamma(q_b[d]) - log_alpha + \
                (q_a[d] - alpha) * (torch.digamma(q_a[d]) - torch.digamma(q_a[d] + q_b[d])) + \
                    (q_b[d] - 1) * (torch.digamma(q_b[d]) - torch.digamma(q_a[d] + q_b[d]))
        
        return kl
    
    def compute_expected_KL_Z(self):
        exp_log_1_minus_pi = torch.zeros((self.D, ), dtype=float_type)
        q_a = torch.exp(self.q_a_raw)
        q_b = torch.exp(self.q_b_raw)
        for d in range(self.D):
            q_d = torch.zeros((d+1, ), dtype=float)
            for i in range(d+1):
                q_d[i] = torch.exp(
                    torch.digamma(q_b[i]) + 
                    torch.sum(torch.digamma(q_a[:i])) - 
                    torch.sum(torch.digamma(q_a[:(i+1)] + q_b[:(i+1)]))
                )
            q_d = q_d + 1e-20
            q_d = q_d / torch.sum(q_d) # TODO: do we need add a small jitter?
            
            for i in range(d+1):
                if i < d:
                    exp_log_1_minus_pi[d] = exp_log_1_minus_pi[d] + (
                        q_d[i] * torch.digamma(q_b[i]) + 
                        torch.sum(q_d[(i+1):]) * torch.digamma(q_a[i]) - 
                        torch.sum(q_d[i:]) * torch.digamma(q_a[i] + q_b[i])
                    )
                else:
                    exp_log_1_minus_pi[d] = exp_log_1_minus_pi[d] + (
                        q_d[i] * torch.digamma(q_b[i]) -
                        torch.sum(q_d[i:]) * torch.digamma(q_a[i] + q_b[i])
                    )
            
            exp_log_1_minus_pi[d] = exp_log_1_minus_pi[d] - torch.sum(torch.log(q_d) * q_d)
            
        exp_log_pi = torch.zeros((self.D, ), dtype=float_type)
        for d in range(self.D):
            exp_log_pi[d] = torch.sum(torch.digamma(q_a[:(d+1)]) - torch.digamma(q_a[:(d+1)] + q_b[:(d+1)]))
        
        q_tau = torch.sigmoid(self.q_tau_logit)
        
        exp_log_q = torch.sum(q_tau * torch.log(q_tau) + (1 - q_tau) * torch.log(1 - q_tau))
        exp_log_p = torch.sum(q_tau * exp_log_pi[None, :, None] + (1 - q_tau) * exp_log_1_minus_pi[None, :, None])
        
        kl = exp_log_q - exp_log_p
        
        return kl
    