import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .components import create_mlp, create_mlp_with_dropout, process_surv, process_clf

class TransformerEmb(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        
        # default
        heads = 8
        dim_head = 64
        dropout = 0.1
        dim = config.in_dim
        self.mode = mode
        self.config = config
        self.n_classes = config.n_classes

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.in_dim), 
                                      requires_grad=True)
        
        self.pre_ln = nn.Linear(dim, dim, bias=False)
        self.pre_norm = nn.LayerNorm(dim)
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = nn.Parameter(torch.randn(1), requires_grad=True)

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.classifier = nn.Linear(config.in_dim, config.n_classes, bias=False)

    def attn_forward(self, x_qkv, mask=None):
        x_qkv = self.pre_ln(x_qkv)
        x_qkv = self.pre_norm(x_qkv)
        
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        if mask is not None:
            attn = attn * \
                mask.long().unsqueeze(1).repeat(1, self.heads, 1, 1)
        # print(attn.shape)
        
        # print(attn.shape, v.shape)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out =  self.to_out(out)
        return out
    
    def forward_no_loss(self, h, attn_mask=None):
        if attn_mask is not None: 
            attn_mask = attn_mask.bool()
        
        h = self.attn_forward(h, attn_mask)
        # h = h[:, 0, :]   # CLS token only
        h = torch.mean(h, dim=1)
        logits = self.classifier(h)
        out = {'logits': logits}
        return out
    
    def forward_no_loss_cls_added(self, h, attn_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
        
        # Add learnable CLS token at the first position
        # cls_tokens = torch.sum(h, dim=1, keepdims=True)
        cls_tokens = self.cls_token.repeat(h.shape[0], 1, 1)
        h = torch.cat([cls_tokens, h], dim=1)  # shape: [batch_size, seq_len+1, hidden_dim]
        
        # Adjust attention mask if provided
        if attn_mask is not None:
            batch_size, seq_len, _ = attn_mask.shape
            # Create new attention mask with size [batch_size, seq_len+1, seq_len+1]
            new_attn_mask = torch.zeros((batch_size, seq_len+1, seq_len+1), 
                                    dtype=torch.bool, 
                                    device=attn_mask.device)
            
            # CLS token can attend to everything (including itself)
            new_attn_mask[:, 0, :] = True
            
            # Existing tokens can attend to CLS token and what they could attend to before
            new_attn_mask[:, 1:, 0] = True  # all tokens can attend to CLS
            new_attn_mask[:, 1:, 1:] = attn_mask  # original attention patterns
            
            attn_mask = new_attn_mask
        
        h = self.attn_forward(h, attn_mask)
        h = h[:, 0, :]   # CLS token only
        logits = self.classifier(h)
        out = {'logits': logits}
        return out
    
    def forward(self, h, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            if h.ndim==2:   # PANTHER
                b = h.size(0)
                h = h.reshape(b, self.p, -1)
            
            out = self.forward_no_loss_cls_added(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")
        
        return results_dict, log_dict
    
class TransformerMLPEmb(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        
        # default
        heads = 8
        dim_head = 64
        dropout = 0.1
        dim = config.in_dim
        self.mode = mode
        self.config = config
        self.n_classes = config.n_classes
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.in_dim), 
                                      requires_grad=True)
        self.pre_norm = nn.LayerNorm(dim)
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # self.scale = dim_head ** -0.5
        self.scale = nn.Parameter(torch.randn(1), requires_grad=True)
        
        self.pre_ln = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.to_k = nn.Sequential(
            nn.Linear(dim, inner_dim , bias=False), nn.ReLU(),
            nn.Linear(inner_dim, inner_dim, bias=False))
        self.to_v = nn.Sequential(
            nn.Linear(dim, inner_dim , bias=False), nn.ReLU(),
            nn.Linear(inner_dim, inner_dim, bias=False))
        self.to_q = nn.Sequential(
            nn.Linear(dim, inner_dim , bias=False), nn.ReLU(),
            nn.Linear(inner_dim, inner_dim, bias=False))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(config.in_dim, config.in_dim // 2, bias=False), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(config.in_dim // 2, config.n_classes, bias=False))
        # self.classifier = nn.Linear(config.in_dim, config.n_classes, bias=False)

    def attn_forward(self, x_qkv, mask=None):
        x_qkv = self.pre_ln(x_qkv)
        x_qkv = self.pre_norm(x_qkv)
        
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        if mask is not None:
            mask = mask.long().unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots = dots * mask + torch.ones_like(dots) * -1e-9 * (1 - mask)
                
                
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        out =  self.to_out(out)
        return out
    
    def forward_no_loss(self, h, attn_mask=None):
        if attn_mask is not None: 
            attn_mask = attn_mask.bool()
        
        h = self.attn_forward(h, attn_mask)
        h = torch.mean(h, dim=1)
        logits = self.classifier(h)
        out = {'logits': logits}
        return out
    
    def forward_no_loss_cls_added(self, h, attn_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
        
        cls_tokens = self.cls_token.repeat(h.shape[0], 1, 1)
        h = torch.cat([cls_tokens, h], dim=1)  # shape: [batch_size, seq_len+1, hidden_dim]
        
        # Adjust attention mask if provided
        if attn_mask is not None:
            batch_size, seq_len, _ = attn_mask.shape
            # Create new attention mask with size [batch_size, seq_len+1, seq_len+1]
            new_attn_mask = torch.zeros((batch_size, seq_len+1, seq_len+1), 
                                    dtype=torch.bool, 
                                    device=attn_mask.device)
            
            # CLS token can attend to everything (including itself)
            new_attn_mask[:, 0, :] = True
            
            # Existing tokens can attend to CLS token and what they could attend to before
            new_attn_mask[:, 1:, 0] = True  # all tokens can attend to CLS
            new_attn_mask[:, 1:, 1:] = attn_mask  # original attention patterns
            
            attn_mask = new_attn_mask
        
        h = self.attn_forward(h, attn_mask)
        h = h[:, 0, :]   # CLS token only
        logits = self.classifier(h)
        out = {'logits': logits}
        return out
    
    def forward(self, h, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']

            if h.ndim==2:   # PANTHER
                b = h.size(0)
                # prob = h[:, : self.p]
                # mean = h[:, self.p: self.p * ( 1+ d )].reshape(-1, self.p, d)
                # cov = h[:, self.p * (1 + d):].reshape(-1, self.p, d)
                h = h.reshape(b, self.p, -1)
            
            out = self.forward_no_loss_cls_added(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']

            out = self.forward_no_loss(h, attn_mask=attn_mask)
            logits = out['logits']

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Not Implemented!")
        
        return results_dict, log_dict