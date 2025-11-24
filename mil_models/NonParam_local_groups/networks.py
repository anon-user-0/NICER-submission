import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# import matplotlib.pyplot as plt
import copy
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from scipy.optimize import linear_sum_assignment

from torch.nn.modules.utils import _pair
from functools import reduce
from operator import mul

from utils.file_utils import load_pkl
from .utils import get_sequential_tensor, get_batched_tensor


class ParallelNonparametricAgg(nn.Module):
    def __init__(self, prompt_dim, n_hidden=32):
        super(ParallelNonparametricAgg, self).__init__()
        self.cov_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, prompt_dim),
            nn.Sigmoid()
        )
        self.bernoulli_net = nn.Sequential(
            nn.Linear(prompt_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

    def prompt_likelihood(self, local_prompts, centroids, z):
        """
        args:
        - local_prompts: n_client x n_local x H
        - centroids: n_global x H
        - z: n_client x n_local x n_global
        """
        _z = torch.tensor(z).to(local_prompts.device) 
        mean_i = centroids  # n_global x H1
        cov_i = self.cov_net(mean_i)    # n_global x H2
        prompt_dist = Independent(Normal(mean_i, cov_i),1)
        lp = prompt_dist.log_prob(local_prompts)    
        
        log_prob = _z * lp 
        # import pdb; pdb.set_trace()
        return log_prob.sum(), lp.unsqueeze(-1) 

    def z_likelihood(self, centroids, z):
        """
        args:
        - centroids: n_global x h
        - z: n_client x n_local x n_global
        """
        _z = torch.tensor(z).to(centroids.device)
        c = torch.sum(_z, dim=1, keepdims=True) 
        lik = 0.
        cost_mat = []
        
        
        prob_i = self.bernoulli_net(centroids) # 1 x n_global 
        
        prompt_dist = Independent(Bernoulli(prob_i),1)
        
        cost_mat = prompt_dist.log_prob(c * torch.ones(_z.shape).to(c)) 
        
        log_prob = _z * cost_mat  
        lik = lik + log_prob.sum()
        return lik, cost_mat.unsqueeze(-1)

    
    def forward(self, local_prompts, outer_loop):
        torch.set_grad_enabled(True)
        
        n_clients, n_local = local_prompts.shape[0], local_prompts.shape[1] # N_max, 1
        
        n_global = n_clients * n_local
        
        # Initialize the output array
        z = np.zeros((n_clients, n_local, n_global))

        perms = np.array([np.random.permutation(n_global) for _ in range(n_clients)])

        for i in range(n_clients):
            z[i, :n_local, perms[i, :n_local]] = 1
        

        centroids = nn.ParameterList([copy.deepcopy(local_prompts.flatten(0, 1))]) # (n_clients x n_prompts) x 768
        
        opt = Adam([
            {'params': self.cov_net.parameters()},
            {'params': self.bernoulli_net.parameters()},
            {'params': centroids}
        ])

        # Alternate opt phi, z
        for i in range(outer_loop):
                
            # solve for all parameters
            opt.zero_grad()
            # Compute l1, l2
            l1, m1 = self.prompt_likelihood(local_prompts, centroids[0], z)
            l2, m2 = self.z_likelihood(centroids[0], z)
            
            loss = -l1 -l2
            
            loss.backward()

            opt.step()

            # Solve for z
            M = m1 + m2
            
            for t in range(n_clients):
                m = M[t,...].t().clone().detach().cpu().numpy()
                # print(m.shape)
                row_id, col_id = linear_sum_assignment(m, maximize=True)
                z[t] *= 0
                z[t][row_id, col_id] += 1
                
        z = np.stack(z)
        z_ = z
        z = np.sum(np.stack(z), axis=(0, 1), keepdims=False) # n_local x n_global
        global_prompts = centroids[0][np.where(z > 0)[0]]
        del z, centroids
        return global_prompts, z_
    
class PrototypePool(nn.Module):
    def __init__(self, embed_dim=128, pool_size=100, top_k=3, dropout_value=0.0, patch_size=16):
        super(PrototypePool, self).__init__()
        patch_size_pair = _pair((patch_size, patch_size))
        self.top_k = top_k
        self.pool_size = pool_size
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim), requires_grad=True)
        self.features_proj = nn.Linear(embed_dim, embed_dim)
       
        self.prompt_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.features_dropout = nn.Dropout(dropout_value)

        # Prompt initialization (uniform distribution)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size_pair, 1) + embed_dim))
        nn.init.uniform_(self.prompt.data, -val, val)
        
    def prompt_projection(self, prompts):
        """
        args:
        - prompts [n_protos x H]
        """
        projected_prompt = self.features_proj(self.features_dropout(prompts))
        return projected_prompt

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, top_k, instance_wise=False, cls_features=None):
        top_k = min(top_k, self.top_k, self.prompt.data.shape[0])
        current_pool_size = self.prompt.shape[0]
        x_embed_mean = x_embed
        
        projected_prompt = self.features_proj(self.features_dropout(self.prompt))
       
        prompt_norm = self.l2_normalize((projected_prompt), dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C=768  
    
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
        _, idx = torch.topk(similarity, k=top_k, dim=1) # B, top_k
        self.top_k_idx_full = idx
        if not instance_wise:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)     
            if prompt_id.shape[0] < current_pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((current_pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])    # current_pool_size=27  
                # [ 1,  2,  4,  7,  8,  9, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1]
                id_counts = torch.cat([id_counts, torch.full((current_pool_size - id_counts.shape[0],), 0, device=id_counts.device)])                           # current_pool_size=27  
                # [284,   1, 511,  44,   4,   1, 511,   2, 170, 511,  14, 240,   4,   5, 168,  81,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
            _, major_idx = torch.topk(id_counts, k=top_k) # top_k most  frequently exists in id_counts      
            # [511, 511, 511, 284, 240], [ 9,  6,  2,  0, 11]
            major_prompt_id = prompt_id[major_idx] # top_k      
            idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            self.top_k_idx = idx[0]
        else:
            self.top_k_idx = idx
        
        batched_prompt = projected_prompt[idx] # B, top_k, C
        batched_key_norm = prompt_norm[idx]
        
        x_embed_norm = x_embed_norm.unsqueeze(1)
        
        sim = batched_key_norm * x_embed_norm   # BxNxK (b=1)
        reduce_sim = - torch.sum(sim) / x_embed.shape[0]
        
        return reduce_sim, batched_prompt

    def get_qq(self, X, proto, sampling=0):
        
        device = X.device
        X = X.to(device)
        N, d = X.shape
        if sampling > 0:
            xs = get_sequential_tensor(X, sampling)
        elif sampling==0 or sampling >= N:
            xs = [X]
            
        clusters = list()
        self.eval()
        for x in xs:
            x = x.to(device)
            projected_prompt = self.features_proj(self.features_dropout(proto))
            prompt_norm = self.l2_normalize(projected_prompt, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x, dim=1) # B, C=768  
            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            # get global cluster idx
            clusters.append(similarity.cpu())
            # clusters.extend(cluster_labels.tolist())
        clusters = torch.cat(clusters, dim=0)
        return clusters

class PrototypeRetrieval(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.pool = PrototypePool(**kwargs)
        self.retrieved_proto_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
        
    def checking_retrieved_prototypes(self, full=False):
        chosen_top_k = self.pool.top_k_idx if not full else self.pool.top_k_idx_full
        if not full:
            chosen_top_k = self.pool.top_k_idx
        else:
            chosen_top_k = self.pool.top_k_idx
            
        if chosen_top_k != self.retrieved_proto_checklist.device:
            self.retrieved_proto_checklist = self.retrieved_proto_checklist.to(self.pool.top_k_idx.device)
        
        flattened_indices = chosen_top_k.view(-1)
        values = torch.ones_like(flattened_indices, dtype=self.retrieved_proto_checklist.dtype)
        self.retrieved_proto_checklist.index_add_(0, flattened_indices, values)
        
    def get_all_retrieved_prototypes(self):
        return self.pool.prompt[torch.nonzero(self.retrieved_proto_checklist).flatten()].clone()
    
    def get_prototypes_by_indices(self, proto_indices):
        selected_prompts = self.pool.prompt.data[proto_indices.tolist()]    # n_proto x H
        self.pool.eval()
        with torch.no_grad():
            proj_prompt = self.pool.prompt_projection(selected_prompts)
        
        return  selected_prompts, proj_prompt
        
    def reset_retrieved_prototypes_checklist(self):
        self.retrieved_proto_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
        
    def reinit_prototype_set(self, prototypes):
        device = self.pool.prompt.device
        # prototypes = prototypes.type(type(self.pool.prompt.cpu().clone().detach().data))
        self.pool.prompt = nn.Parameter(prototypes, requires_grad=True)
        self.pool.pool_size = prototypes.shape[0]
        self.reset_retrieved_prototypes_checklist()
        return
        
    def forward(self, X, top_k=3, instance_wise=False, sampling=0):
        
        device = X.device
        X = X.cpu()
        N, d = X.shape
        
        x = X.to(device)
        reduced_sim, retrieved_protos = self.pool(x, top_k, instance_wise)
        self.checking_retrieved_prototypes()
        
        return reduced_sim, retrieved_protos
    
    def get_qq(self, x, proto):
        return self.pool.get_qq(x, proto)
    
    def train_similarity(self, x, n_steps=5, train_lin=False, 
                         instance_wise=True, sampling=0, lr=0.1, 
                         top_k=3):
        self.pool.train()
        parameters = [pr for n, pr in self.pool.named_parameters() if 'proj' not in n]
        if train_lin:
            parameters = self.pool.parameters()
        opt = Adam([{
            "params": parameters
        }], lr=lr)
        
        N, d = x.shape
        xs = get_batched_tensor(x, sampling_num=sampling) if (sampling > 0 and sampling < N) \
            else [x]
        
        losses = list()
        with torch.set_grad_enabled(True):
            for i in range(n_steps):
                for xb in xs:
                    xb = xb.reshape(-1, xb.size(-1))
                    opt.zero_grad()
                    sim_loss, _ = self.forward(xb, top_k=top_k, 
                                               instance_wise=instance_wise)
                    # sim_loss.requires_grad = True
                    losses.append(sim_loss.detach().cpu())
                    sim_loss.backward()
                    opt.step()
            
        return losses
    
    def infer_similarity(self, x, instance_wise=True, sampling=0, top_k=3):
        self.pool.eval()
        N, d = x.shape
        xs = get_sequential_tensor(x, sampling_num=sampling) if (sampling > 0 and sampling < N) \
            else [x]
        
        losses = list()
        with torch.set_grad_enabled(True):
            for xb in xs:
                xb = xb.reshape(-1, xb.size(-1))
                sim_loss, _ = self.forward(xb, top_k=top_k, 
                                           instance_wise=instance_wise)
                # sim_loss.requires_grad = True
                losses.append(sim_loss.detach().cpu())
                sim_loss.backward()
            
        return losses