from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .constants import float_type


class Kernel(nn.Module):
    def __init__(self, input_dim: int, active_dims=None):
        super().__init__()
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if (active_dims.start is not None) and (active_dims.stop is not None) and (active_dims.step is not None):
                assert len(range(*active_dims)) == input_dim
        else:
            self.active_dims = torch.from_numpy(np.array(active_dims, dtype=int))
            assert len(active_dims) == input_dim
        
        self.num_gauss_hermite_points = 20
        
    def _slice(self, x: torch.Tensor, x2: Optional[torch.Tensor]):
        if isinstance(self.active_dims, slice):
            x = x[:, self.active_dims]
            if x2 is not None:
                x2 = x2[:, self.active_dims]
        else:
            # TODO: make sure that torch gather is used correctly!
            x = torch.gather(torch.transpose(x, 0, 1), dim=0, index=self.active_dims)
            if x2 is not None:
                x2 = torch.gather(torch.transpose(x2, 0, 1), dim=0, index=self.active_dims)
        
        return x, x2
    
    def _slice_cov(self, cov: torch.Tensor):
        N = cov.shape[0]
        if len(cov.shape) == 2:
            cov = torch.cat([torch.diag(cov[n])[None] for n in range(N)], dim=0)
        if isinstance(self.active_dims, slice):
            cov = cov[:, self.active_dims, self.active_dims]
        else:
            cov = torch.gather(cov, dim=1, index=self.active_dims)
            cov = torch.gather(cov, dim=2, index=self.active_dims)
        
        return cov
    
    
class Stationary(Kernel):
    def __init__(
        self, 
        input_dim: int,
        variance: float = 1.0, 
        lengthscales: float = 1, 
        active_dims=None, 
    ):
        super().__init__(input_dim=input_dim, active_dims=active_dims)
        self.lengthscales = nn.Parameter(torch.tensor(lengthscales, dtype=float_type), requires_grad=True)
        self.variance = nn.Parameter(torch.tensor(variance, dtype=float_type), requires_grad=True)
        
    def square_distance(self, x: torch.Tensor, x2: Optional[torch.Tensor]):
        x = x / self.lengthscales
        x_sum = torch.sum(torch.square(x), axis=1)
        
        if x2 is None:
            return -2 * torch.matmul(x, torch.transpose(x, -2, -1)) + x_sum[:, None] + x_sum[None]
        else:
            x2 = x2 / self.lengthscales
            x2_sum = torch.sum(torch.square(x2), axis=1)
            return -2 * torch.matmul(x, torch.transpose(x2, -2, 1)) + x_sum[:, None] + x2_sum[None]
    
    def euclid_distance(self, x: torch.Tensor, x2: Optional[torch.Tensor]):
        r2 = self.square_distance(x, x2)
        return torch.sqrt(r2 + 1e-12)
    
    def K_diagonal(self, x: torch.Tensor, presliced: bool=False):
        return torch.ones((x.shape[0], ), device=x.device) * torch.square(self.variance)
    
    
class SquaredExponential(Stationary):
    def __init__(self, input_dim: int, variance: float = 1.0, lengthscales: float = 1.0, active_dims=None):
        super().__init__(input_dim=input_dim, variance=variance, lengthscales=lengthscales, active_dims=active_dims)

    def K(self, x: torch.Tensor, x2: Optional[torch.Tensor]=None, presliced: bool=False):
        if not presliced:
            x, x2 = self._slice(x, x2)
        out = torch.square(self.variance) * torch.exp(-self.square_distance(x, x2) / 2)
        
        return out
    