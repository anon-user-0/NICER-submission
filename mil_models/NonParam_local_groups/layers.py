import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from .networks import DirNIWNet, NonparametricAgg, PrototypeRetrieval, ParallelNonparametricAgg
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.file_utils import load_pkl
from .utils import save2pkl

class NonParamBaseIter(nn.Module):
    """
    Args:
    - p (int): Number of prototypes
    - d (int): Feature dimension
    - L (int): Number of EM iterations
    - out (str): Ways to merge features
    - ot_eps (float): eps
    """
    def __init__(self, d, p=5, L=3, tau=10.0, out='allcat', ot_eps=0.1,
                 load_proto=True, proto_path='.', fix_proto=True, compression_path='.',
                 similarity_lr=0.1, top_k=3):
        super(NonParamBaseIter, self).__init__()

        self.L = L
        self.tau = tau
        self.out = out
        self.H = 1  # This is a dummy variable - Originally intended for multihead. Always keep it at 1
        self.pca_dim = 128
        self.d = d
        self.similarity_lr = similarity_lr
        self.top_k = top_k
        
        self.prev_mu = None
        self.reinit_retriever()
        
        if out == 'allcat':  # Concatenates pi, mu, cov - This is the default mode for PANTHER used in the paper
            self.outdim = p + 2*p*d

        elif out == 'weight_all_cat':   # Weights mu and cov by weights and that concatenate
            self.outdim = 2 * p * d

        elif 'select_top' in out:   # Select top c components
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto

        elif 'select_bot' in out:   # Select bottom c components
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            assert numOfproto <= p
            self.outdim = numOfproto * 2 * d + numOfproto

        elif out == 'weight_avg_all':   # Concatenates mu and cov weighted by pi
            self.outdim = 2 * d

        elif out == 'weight_avg_mean':   # Concatenates mu weighted by pi
            self.outdim = d
        else:
            raise NotImplementedError("Out mode {} not implemented".format(out))
        
    def reinit_retriever(self):
        kwargs = {
            "embed_dim": 1024,
            "pool_size": 200,
            "top_k": 3,
            "dropout_value": 0.1
        }
        self.retriever = PrototypeRetrieval(kwargs)
        self.retriever.train()
        
        return
    
    def load_state_dict(self, state_dict):
        self.retriever.reinit_prototype_set(state_dict["retriever.pool.prompt"])
        super().load_state_dict(state_dict)
        return
        
    def unsup_train_single_step(self, S, mu=None, n_steps=1,
                                train_lin=False, instance_wise=True,
                                sampling=0, idx=0):
        """
        Args
        - S: data
        """
        
        S = S[0]
        device = S.device
        S = S.cpu()
        N, d = S.shape
        
        torch.manual_seed(42)
        # Define split ratio (e.g., 70% training, 30% validation)
        train_ratio = 0.8
        n_train = int(train_ratio * N)  # Number of training samples
        
        # Generate random indices and shuffle
        indices = torch.randperm(N)  # Random permutation of indices [0, N-1]

        # Split indices into training and validation
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create training and validation sets
        S_train = S[train_indices]  # Training set
        S_val = S[val_indices]    # Validation set

        S_train = S_train.to(device)
        S_val = S_val.to(device)
        
        initial_s = torch.randn(sampling, S_train.shape[-1], device=device) if sampling > 0 else S_train
        
        self.reinit_retriever()
        
        self.retriever.reinit_prototype_set(initial_s)
        # self.retriever.reinit_prototype_set(s)
        self.retriever.to(device)

        iter_num = 10
        train_losses, valid_losses, prototypes, assignments = list(), list(), list(), list()
        mus = list()
        
        prototypes.append(S.cpu().detach().numpy())
        prototypes.append(initial_s.cpu().detach().numpy())
        
        best_valid_loss = 1e9
        for it in range(iter_num):
            S_train = S_train.to(device)
            union_prototypes_checklist = torch.zeros(self.retriever.pool.prompt.shape[0], dtype=torch.float32)
            
            losses = self.retriever.train_similarity(S_train, n_steps=n_steps,
                                                    train_lin=train_lin, instance_wise=instance_wise,
                                                    sampling=sampling, lr=self.similarity_lr, top_k = self.top_k)
            
            self.retriever.checking_retrieved_prototypes(full=True) # regardless of similarity training type, get full relevant prompts for aggregation
            mu = self.retriever.pool.prompt.data.clone()
            nonzero_index = torch.nonzero(self.retriever.retrieved_proto_checklist).flatten()
            
            union_prototypes_checklist[nonzero_index] = 1

            self.retriever.reset_retrieved_prototypes_checklist()
            
            mu = mu[torch.nonzero(union_prototypes_checklist).flatten()].clone()
            mu = torch.stack([mu], dim=0)    # Bxn_protoxd
            B, n_proto, d = mu.shape
            mu = mu.permute(1,0,2)
            aggr= ParallelNonparametricAgg(self.d, n_hidden=128).to(device)
            summary_mu, z = aggr(mu, 5)
            assignments.append(z.squeeze(1))
            
            self.retriever.reinit_prototype_set(summary_mu)
            # if it < iter_num-1: del summary_mu
            
            train_losses.append(np.mean(losses))
                                                            
            curr_valid_loss = np.mean(self.retriever.infer_similarity(S_val, instance_wise=instance_wise,
                                                                    sampling=sampling, top_k=self.top_k))
            valid_losses.append(curr_valid_loss)
            
            if curr_valid_loss < best_valid_loss:
                best_valid_loss = curr_valid_loss  # Update best validation loss
                torch.save(self.state_dict(), './tmp_state_dict_quant.pth')  # Save model state

            prototypes.append(summary_mu.cpu().detach().numpy())
            mus.append(mu.cpu().detach().numpy())
            
        data = {
            "prototypes": prototypes,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "assignments": assignments,
            "mus": mus
        }
        
        return summary_mu, data

    def unsup_train_complete(self, data_loader, use_cuda=True):
            
        print("[CKPT 1] Done unsup train complete")
        return
        
    def forward(self, S, mask=None, sampling=0, idx=0):
        """
        Args
        - S: data
        """
        B, N_max, d = S.shape
        
        self.retriever.train()
        _, data = self.unsup_train_single_step([S.squeeze(0)], n_steps=1,
                                        train_lin=True, instance_wise=True,
                                        sampling=sampling, idx=idx)
        
        self.load_state_dict(torch.load('./tmp_state_dict_quant.pth', map_location=S.device))
        
        mus = []
        self.retriever.reset_retrieved_prototypes_checklist()
        
        self.retriever.eval()
        
        
        _, prototypes = self.retriever.forward(S.squeeze(0), top_k=3, instance_wise=True)    # prototype: N_max x top_k x H
        retrieved_counts = self.retriever.retrieved_proto_checklist[
                                    torch.nonzero(self.retriever.retrieved_proto_checklist).flatten()]
        
        count, proto_indices = torch.sort(retrieved_counts, descending=True)
        
        pi = count / torch.sum(count)
        pis = [pi]
        
        self.retriever.eval()
        selected_protos, projected_protos = self.retriever.get_prototypes_by_indices(proto_indices)
        data["selected_protos"] = selected_protos.cpu().detach().numpy()
        data["projected_protos"] = projected_protos.cpu().detach().numpy()
        
        with torch.no_grad():
            qqs = self.retriever.get_qq(S.squeeze(0), selected_protos)
        
        retrieved_prototypes = [torch.cat([selected_protos, projected_protos], dim=-1)]
        protos = torch.stack(retrieved_prototypes, dim=-1).unsqueeze(0) # mus: (n_batch x n_proto x instance_dim x n_head)
        pis = torch.stack(pis, dim=-1).unsqueeze(0) # pis: (n_batch x n_proto x n_head)
        pis = pis.to(protos)

        if self.out == 'allcat':

            protos = protos.squeeze(-1)
            out = torch.cat([pis, protos], dim=-1)   # batch_size x n_protos x (2d + 1)
            
        elif self.out == 'weight_avg_mean':
            """
            Take weighted average of mu according to estimated pi
            """
            out = []
            for h in range(self.H):
                pi, proto = pis[..., h].reshape(B, 1, -1), protos[..., h]
                proto_weighted = torch.bmm(pi, proto).squeeze(dim=1)  # (n_batch, instance_dim)

                out.append(proto_weighted)

            out = torch.cat(out, dim=1) # 1024

        elif 'select_top' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        elif 'select_bot' in self.out:
            c = self.out[10:]
            numOfproto = 1 if c == '' else int(c)

            out = []
            for h in range(self.H):
                pi, mu, Sigma = pis[..., h], mus[..., h], Sigmas[..., h]
                _, indices = torch.topk(-pi, numOfproto, dim=1)

                out.append(pi[:, indices].reshape(pi.shape[0], -1))
                out.append(mu[:, indices].reshape(mu.shape[0], -1))
                out.append(Sigma[:, indices].reshape(Sigma.shape[0], -1))
            out = torch.cat(out, dim=1)

        else:
            raise NotImplementedError

        return out, qqs
