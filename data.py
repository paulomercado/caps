import pandas as pd
import numpy as np
from functools import reduce
import pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, lag=2):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.lag = lag
    
    def __len__(self):
        return len(self.X) - self.lag
    
    def __getitem__(self, idx):
        # pick only the lagged observation(s)
        return self.X[idx:idx+self.lag:self.lag], self.y[idx+self.lag]