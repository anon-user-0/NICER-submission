import torch
import numpy as np
import math

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .constants import float_type


def gaussian_log_pdf(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor):
    return -0.5 * torch.log(2 * math.pi) - 0.5 * torch.log(var) - 0.5 * torch.square(mu - x) / var


def bernoulli_log_pdf(x: torch.Tensor, p: torch.Tensor):
    return torch.log(torch.where(x == 1, p, 1-p))


def poisson_log_pdf(x: torch.Tensor, lmd: torch.Tensor):
    return x * torch.log(lmd) - lmd - torch.lgamma(x + 1.)


def exponential_log_pdf(x: torch.Tensor, lmd: torch.Tensor):
    return -x / lmd - torch.log(lmd)


def multivariate_normal_log_pdf(x: torch.Tensor, mu: torch.Tensor, L: torch.Tensor):
    # L is the Cholesky decomposition of the covariance matrix
    d = x - mu
    alpha = torch.triangular_solve(d, L, upper=False) # lower triangular
    D = x.shape[0]
    N = 1 if len(x.shape) == 0 else x.shape[1]
    ret = -0.5 * N * D * torch.log(2 * math.pi) - torch.sum(torch.log(torch.diagonal(L))) - 0.5 * torch.sum(torch.square(alpha))
    
    return ret
