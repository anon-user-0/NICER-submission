import os
import pickle
import torch
from torch.utils import data

def get_sampled_tensor(S, sampling_num):
    N, d = S.shape
    # Randomly sample 10,000 indices
    indices = torch.randint(0, N, (sampling_num,))

    # Sample the tensor using these indices
    s = S[indices]
    return s

class Splitter(data.Dataset):
    def __init__(self, X):
        super().__init__()
        self.X = X
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx:idx+1, ...]
    
def get_batched_tensor(S, sampling_num):
    N, d = S.shape
    if sampling_num >= N: return S
    else:
        dataset = Splitter(S)
        dataloader = data.DataLoader(dataset, batch_size=sampling_num, shuffle=True)
            
    return dataloader

def get_sequential_tensor(S, sampling_num):
    N, d = S.shape
    if sampling_num >= N: return S
    else:
        samplings = list()
        for idx in range(N // sampling_num + 1):
            start = idx * sampling_num
            end = min((idx + 1)*sampling_num, N)
            s = S[start : end, :]
            samplings.append(s)
            
    return samplings

def save2pkl(data, pkl_path):
    folder = os.path.dirname(pkl_path)
    os.makedirs(folder, exist_ok=True)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)