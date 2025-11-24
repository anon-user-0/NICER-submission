import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .constants import float_type


class MeanFunction:
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError("The __call__ method is not implemented for this mean function!")


class ZeroMeanFunction(MeanFunction):
    def __call__(self, x: torch.Tensor):
        return torch.zeros((x.shape[0], 1), dtype=float_type, device=x.device)
    