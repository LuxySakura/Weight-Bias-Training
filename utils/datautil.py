import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PBL_Dataset(Dataset):
    def __init__(self, filename, repeat = 1):
        dat = np.loadtxt(filename, delimiter = ',', dtype = float, encoding = 'utf-8-sig')
        dat /= dat[0]
        dat = dat.T
        self.data = torch.from_numpy(dat)
        self.len = self.data.shape[0]
        self.repeat = repeat
    
    def __getitem__(self, index):
        idx = index % self.len
        return self.data[idx]
    
    def __len__(self):
        if self.repeat == None:
            return 100000000
        return self.len * self.repeat
    
    def Days(self):
        return self.data.shape[1]
