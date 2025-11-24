import numpy as np
import torch

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from .constants import np_float_type

def hermgauss(n: int):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = torch.from_numpy(x.astype(np_float_type)), torch.from_numpy(w.astype(np_float_type))
    
    return x, w
