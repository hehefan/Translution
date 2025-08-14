import os
import sys 
import random
import numpy as np
from torch.utils.data import Dataset

class TrainLangDataset(Dataset):
    def __init__(self, data_path, seq_len, num_samples_per_epoch):

        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.num_samples_per_epoch = num_samples_per_epoch
        self.idx_range = len(self.data) - seq_len - 1 

    def __len__(self):

        return self.num_samples_per_epoch

    def __getitem__(self, _): 
    
        idx = random.randint(0, self.idx_range)
        data = self.data[idx:idx+self.seq_len].astype(np.int64)
        target = self.data[idx+1:idx+1+self.seq_len].astype(np.int64)

        return data, target

class ValLangDataset(Dataset):
    def __init__(self, data_path, seq_len):

        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.idx_range = list(range(0, len(self.data) - seq_len - 1, seq_len))

    def __len__(self):

        return len(self.idx_range)

    def __getitem__(self, idx):
    
        idx = self.idx_range[idx]
        data = self.data[idx:idx+self.seq_len].astype(np.int64)
        target = self.data[idx+1:idx+1+self.seq_len].astype(np.int64)

        return data, target
