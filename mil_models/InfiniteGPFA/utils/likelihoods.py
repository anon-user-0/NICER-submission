from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .densities import (
    gaussian_log_pdf, 
    poisson_log_pdf, 
    bernoulli_log_pdf, 
)
from .quadrature import hermgauss
from .general_utils import probit


class Likelihood(nn.Module):
    def __init__(self, num_hermgauss_points: int = 20):
        super().__init__()
        self.num_hermgauss_points = num_hermgauss_points
    
    def log_p(self, f: torch.Tensor, y: torch.Tensor):
        """
        Return the log density of the data given the latent factor values
        """
        raise NotImplementedError("The log_p function for this likelihood not yet implemented!")
    
    def conditional_mean(self, f: torch.Tensor):
        """
        Compute conditional mean of the observation given the latent factors
        E[y|f] = \int dy [y p(y|f)]
        """
        raise NotImplementedError
    
    def conditional_variance(self, f: torch.Tensor):
        """
        Compute conditional variance of the observation given the latent factors
        Var[y|f] = \int dy [y^2 p(y|f)] - (\int [dy y p(y|f)])^2
        """
        raise NotImplementedError
    
    def predict_mean_var(self, f_mu: torch.Tensor, f_var: torch.Tensor):
        """
        Given the latent Gaussian (process) factors, approximate the predictive mean and variance of observation (Y)
        I.e., given q(f) = N(f_mu, f_var), compute \int\int df dy y p(y|f)q(f) and 
        \int\int df dy y^2 p(y|f)q(f) - [\int\int df dy y p(y|f)q(f)]^2
        
        Here we implement the general-purpose approximate inference leveraging Gauss-Hermite quadrature method.
        However, some conditional likelihood might admit easier computations (e.g., Gaussian) and will be implemented separately.
        """
        
        hg_x, hg_w = hermgauss(self.num_hermgauss_points)
        hg_w = (hg_w / np.sqrt(np.pi))[:, None]

        input_shape = list(f_mu.shape)
        
        f_mu = f_mu.reshape(-1, 1)
        f_var = f_var.reshape(-1, 1)
        x = hg_x[None] * torch.sqrt(2.0 * f_var) + f_mu
        
        # quadrature for the mean
        E_y = torch.matmul(self.conditional_mean(x), hg_w).reshape(input_shape)
        
        # quadrature for the variance
        integrand = self.conditional_variance(x) + torch.square(self.conditional_mean(x))
        V_y = torch.matmul(integrand, hg_w).reshape(input_shape) - torch.square(E_y)
        
        return E_y, V_y
    
    def predict_density(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the log predictive density of the observation, Y.
        Compute \int p(Y=y|f)q(f) df
        Here we implement the general-purpose approximate inference leveraging Gauss-Hermite quadrature method.
        However, some conditional likelihood might admit easier computations (e.g., Gaussian and Poisson) and will 
        be implemented separately.
        """
        hg_x, hg_w = hermgauss(self.num_hermgauss_points)
        hg_w = (hg_w / np.sqrt(np.pi))[:, None]
        
        input_shape = list(f_mu.shape)
        
        f_mu = f_mu.reshape(-1, 1)
        f_var = f_var.reshape(-1, 1)
        x = hg_x[None] * torch.sqrt(2.0 * f_var) + f_mu
        
        y = torch.tile(y, (1, self.num_hermgauss_points))
        
        log_p = self.log_p(x, y)
        
        return torch.log(torch.matmul(torch.exp(log_p), hg_w)).reshape(input_shape)
    
    def variational_expectations(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        """
        Compute the expected log density of teh observation, Y.
        Compute \int \log p(y|f) q(f) df
        Here we implement the general-purpose approximate inference leveraging Gauss-Hermite quadrature method.
        However, some conditional likelihood might admit easier computations (e.g., Gaussian) and will be implemented separately.
        """
        hg_x, hg_w = hermgauss(self.num_hermgauss_points)
        hg_x = hg_x.reshape(1, -1)
        hg_w = hg_w.reshape(-1, 1) / np.sqrt(np.pi)
        
        input_shape = list(f_mu.shape)
        
        f_mu = f_mu.reshape(-1, 1)
        f_var = f_var.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        hg_x = hg_x.to(f_mu.device)
        
        x = hg_x * torch.sqrt(2.0 * f_var) + f_mu
        y = torch.tile(y, (1, self.num_hermgauss_points))
        
        log_p = self.log_p(x, y)
        
        return torch.matmul(log_p, hg_w).reshape(input_shape)
    
    
class Gaussian(Likelihood):
    def __init__(self, variance: float = 1.0):
        super().__init__()
        self.variance = variance
        
    def log_p(self, f: torch.Tensor, y: torch.Tensor):
        return gaussian_log_pdf(y, f, self.variance)
    
    def conditional_mean(self, f: torch.Tensor):
        return f
    
    def conditional_variance(self, f: torch.Tensor):
        return torch.ones_like(f) * self.variance
    
    def predict_mean_var(self, f_mu: torch.Tensor, f_var: torch.Tensor):
        return f_mu, f_var + self.variance
    
    def predict_density(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        return gaussian_log_pdf(y, f_mu, f_var + self.variance)
    
    def variational_expectations(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        diff = y - f_mu
        diff = torch.clamp(diff, min=-10, max=10)
        out = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(self.variance + 1e-6) - \
                0.5 * (torch.square(diff) + f_var) / (self.variance + 1e-6)
        
        return out
            
class Gaussian_with_link(Likelihood):
    def __init__(self, variance: float=1.0, inverse_link: Optional[Mapping]=None):
        super().__init__()
        self.variance = torch.tensor(variance)
        self.inverse_link = inverse_link
        
    def log_p(self, f: torch.Tensor, y: torch.Tensor):
        return gaussian_log_pdf(y, self.inverse_link(f), self.variance)

    def conditional_mean(self, f: torch.Tensor):
        return self.inverse_link(f)
    
    def conditional_variance(self, f: torch.Tensor):
        return torch.ones_like(self.inverse_link(f)) * self.variance
    
    
class Poisson(Likelihood):
    def __init__(self, inverse_link: Mapping=torch.exp):
        super().__init__()
        self.inverse_link = inverse_link
        
    def log_p(self, f: torch.Tensor, y: torch.Tensor):
        return poisson_log_pdf(y, self.inverse_link(f))
    
    def conditional_variance(self, f: torch.Tensor):
        return self.inverse_link(f)
    
    def conditional_mean(self, f: torch.Tensor):
        return self.inverse_link(f)
    
    def variational_expectations(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        # if self.inverse_link is torch.exp:
        #     return y * f_mu - torch.exp(f_mu + f_var / 2) - torch.lgamma(y + 1)
        
        # return super().variational_expectations(f_mu, f_var, y)
        
        # try using second-order Taylor expansion for evaluating the expectation of exp(h_m)
        expectation_exp_h = torch.exp(f_mu) + 0.5 * f_var * torch.exp(f_mu)
        
        return y * f_mu - expectation_exp_h - torch.lgamma(y + 1.0)


class Bernoulli(Likelihood):
    def __init__(self, inverse_link: Mapping=probit):
        super().__init__()
        self.inverse_link = inverse_link
    
    def log_p(self, f: torch.Tensor, y: torch.Tensor):
        return bernoulli_log_pdf(self.inverse_link(f), y)
    
    def conditional_mean(self, f: torch.Tensor):
        return self.inverse_link(f)
    
    def conditional_variance(self, f: torch.Tensor):
        p = self.inverse_link(f)
        return p - torch.square(p)
    
    def predict_mean_var(self, f_mu: torch.Tensor, f_var: torch.Tensor):
        if self.inverse_link is probit:
            p = probit(f_mu / torch.sqrt(1 + f_var))
            return p, p - torch.square(p)
        else:
            return super().predict_mean_var(f_mu, f_var)
    
    def predict_density(self, f_mu: torch.Tensor, f_var: torch.Tensor, y: torch.Tensor):
        p = self.predict_mean_var(f_mu, f_var)[0]
        return bernoulli_log_pdf(y, p)
