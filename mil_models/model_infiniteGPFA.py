# Model initiation for PANTHER

from torch import nn
import numpy as np

from tqdm import tqdm
from .components import predict_surv, predict_clf, predict_emb,\
            predict_clf_svGPFA, predict_surv_svGPFA
from .InfiniteGPFA.models import *
from .InfiniteGPFA.utils.likelihoods import Gaussian_with_link, Gaussian


import torch
torch.autograd.set_detect_anomaly(True)


class InfiniteGPFA(nn.Module):
    """
    Wrapper for NonParam model
    """
    def __init__(self, config, mode):
        super(InfiniteGPFA, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size
        self.load_proto = config.load_proto
        self.mode = mode
        self.p = 16
        
        self.alpha_prior = {
            "s1": 1.0, 
            "s2": 1.0, 
        }
        self.C_prior = None
        self.d_prior = None
        
        self.lmd_orthogonality = 1.0

    def representation(self, Y):
        """
        Construct unsupervised slide representation
        """
        if len(Y.shape) > 2: Y = Y.squeeze(0)   # NxD only
        np.random.seed(42)
        data_size = Y.shape[0]
        bs = int(min(1e4, data_size))
        
        
        X = torch.from_numpy(
            np.random.randn(Y.shape[1]).reshape(-1, 1)).to(Y)
        
        C = torch.randn([bs, self.p], device=Y.device).float()
        d = torch.randn([bs, 1], device=Y.device).float()
        Y = Y.unsqueeze(-1).permute(1,0,2)
        
        Y = Y - torch.mean(Y, dim=1, keepdims=True) / (torch.std(Y, dim=1, keepdims=True) + 1e-6)
        
        # print(Y.shape)
        num_inducings = [30 for _ in range(self.p)]
        Z = [
            torch.from_numpy(np.random.uniform(0, bs, num_inducings[d_])
                             .astype(np.float32).reshape(-1, 1)).to(Y.device)
            for d_ in range(self.p)
        ]
        
        likelihood = Gaussian(variance=0.1)
        
        device = Y.device
        Y = Y.cpu()
        model = None
        
        for e in range(30):
            
            yid = 0
            while yid <= (data_size - bs):
                start = int(yid)
                end = int(yid + bs)
                y_ = Y[:, start: end].to(device)
                
                if model is None:
                    model = InfiniteSparseVariationalGPFA(
                        X=X, 
                        Y=y_, 
                        D=self.p, # number of latent vectors 
                        likelihood=likelihood, 
                        Z=Z, 
                        C=C, 
                        d=d, 
                        q_diagonal=False, 
                        alpha=1.0, 
                        train_alpha=True, 
                        C_prior=self.C_prior, 
                        d_prior=self.d_prior, 
                        alpha_prior=self.alpha_prior, 
                        m_step=True, 
                        train_inducing_locs=False, 
                        lmd_orthogonality=self.lmd_orthogonality, 
                    )
                    model.to(device)
                    model.train()
                    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
                    
                model.train()
                    
                model.Y = y_
                
                optimiser.zero_grad()
                free_energy = model.variational_free_energy()
                free_energy.backward()
                optimiser.step()
                
                yid += bs
        
        with torch.no_grad():
            f_means, f_covs, log_rates_mean, log_rates_var = model.predict_log_rates(Y.to(device))
        f_means = f_means.permute(1,0,2)
        
        return {'repr': f_means, 'qq': None}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def unsup_train_predict(self, data_loader, use_cuda=True):
        if self.mode == 'classification':
            output, mask, y = predict_clf_svGPFA(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, mask, y
    
    def predict(self, data_loader, use_cuda=True, trainable=False):
        
        if trainable:
            print("start unsup training")
            self.nonparam.unsup_train_complete(data_loader, use_cuda)
        
        mask = None
        if self.mode == 'classification':
            output, y = predict_clf_svGPFA(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, y = predict_surv_svGPFA(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, mask, y