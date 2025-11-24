import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size))
    def forward(self, x):
        feats = self.fc(x)
        return feats

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # B x N x K
        # feats = torch.mean(feats, dim=1)
        c = self.fc(feats) # B x N x C
        return feats, c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=False, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(input_size, 128*4), nn.ReLU(),
                nn.Linear(128*4, 128*2), nn.ReLU(),
                nn.Linear(128*2, 128), nn.Tanh()
                )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU(),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # B x N x K, B x N x C
        device = feats.device
        V = self.v(feats) # B x N x V, unsorted
        Q = self.q(feats) # B x N x Q, unsorted
        
        # handle multiple classes without for loop
        # _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        # m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        
        values, m_indices = torch.topk(c, k=1, dim=1)

        # m_indices is [B, 1, C], we want [B, C]
        m_indices = m_indices.squeeze(1)   # [B, C]

        # Gather features corresponding to those indices
        # feats: [B, N, K]
        # Need to expand indices to match feats’ shape for gather
        B_, N, K = feats.shape
        _, C_ = m_indices.shape

        # Expand indices: [B, C, K]
        expanded_idx = m_indices.unsqueeze(-1).expand(-1, -1, K)

        # Gather features per batch, per class
        m_feats = torch.gather(feats, dim=1, index=expanded_idx)  # [B, C, K]
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.bmm(Q, q_max.transpose(1, 2)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[2], dtype=torch.float32, device=device)), 1) # normalize attention scores, A in shape N x C, 
        B = torch.bmm(A.transpose(1, 2), V) # compute bag representation, B in shape C x V
                
        B = B.view(-1, B.shape[1], B.shape[2]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(B_, -1)
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, config, mode):
        super(DSMIL, self).__init__()
        self.config = config
        self.mode = mode
        self.in_dim = 1024
        # self.in_dim = config.in_dim
        self.i_classifier = IClassifier(
            feature_extractor=FCLayer(in_size=self.in_dim,
                                      out_size=config.embed_dim),
            feature_size=config.embed_dim,
            output_class=config.n_classes
        )
        self.b_classifier = BClassifier(
            input_size=config.embed_dim,
            output_class=config.n_classes,
            nonlinear=True, passing_v=True
        )
        
    def forward_no_loss(self, x):
        # if len(x.shape) == 2:
        #     x = x.unsqueeze(-1).repeat(1, 1, self.in_dim)
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if x.shape[-1] > self.in_dim:
            x = x[:, :, 1: self.in_dim+1]
            
        # if x.size(0)==1:
        #     attn_mat = torch.sum(x.abs(), axis=-1)
        #     attn_idx = torch.nonzero(attn_mat.squeeze()).max()
        #     x = x[:, :attn_idx+1, :]
        
        feats, classes = self.i_classifier(x)
        out, A, B = self.b_classifier(feats, classes)
        
        # return classes, prediction_bag, A, B
        return {'logits': out}
    
    def forward(self, h, model_kwargs={}):
        h = h.float()
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h)
            logits = out['logits']
            
            results_dict, log_dict = process_clf(logits, label, loss_fn)
            
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
            
        else:
            raise NotImplementedError("Not Implemented!")
        
        return results_dict, log_dict    
        
    def forward_ig(self, h):
        out = self.forward_no_loss(h)
        logits = out['logits']
        return logits