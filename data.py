from functools import reduce
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lag_years=2):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = lag_years * 12  # convert years to months

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.seq_len]   # 24 months of input
        y = self.y[idx+self.seq_len]       # target month
        return x, y
