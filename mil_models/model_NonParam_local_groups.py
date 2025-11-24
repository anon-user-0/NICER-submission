# Model initiation for PANTHER

import time
from torch import nn
import numpy as np

from .components import predict_surv_nonparam, predict_clf, predict_emb, predict_clf_nonparam
from .NonParam_local_groups.layers import *
from utils.proto_utils import check_prototypes
from utils.file_utils import load_pkl


class NonParam_Local_Groups(nn.Module):
    """
    Wrapper for NonParam model
    """
    def __init__(self, config, mode):
        super(NonParam_Local_Groups, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.heads = config.heads
        self.outsize = config.out_size
        self.load_proto = config.load_proto
        self.mode = mode
        
        self.nonparam = NonParamBaseIter(self.emb_dim, p=config.out_size, L=config.em_iter,
                         tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                         load_proto=config.load_proto, proto_path=config.proto_path,
                         fix_proto=config.fix_proto, compression_path=config.compression_path,
                         similarity_lr=config.similarity_lr, top_k=config.top_k)

    def representation(self, x, idx=0):
        """
        Construct unsupervised slide representation
        """
        

       
        out, qqs = self.nonparam(x, sampling=self.config.sampling_num, idx=idx)

        torch.cuda.synchronize()  # wait for kernels to finish
        

        return {'repr': out, 'qq': qqs}

    def forward(self, x, idx=0):
        out = self.representation(x, idx=idx)
        return out['repr']
    
    def unsup_train_predict(self, data_loader, use_cuda=True):
        
        if self.mode == 'classification':
            output, mask, y = predict_clf_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, mask, y = predict_surv_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
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
            output, mask, y, qqs = predict_clf_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'survival':
            output, mask, y = predict_surv_nonparam(self, data_loader.dataset, use_cuda=use_cuda)
        elif self.mode == 'emb':
            output = predict_emb(self, data_loader.dataset, use_cuda=use_cuda)
            y = None
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}!")
        
        return output, mask, y, qqs