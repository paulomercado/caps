from functools import reduce
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len 
        
    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len])

def transform_data(data, save_path="Transforms/default/scaler.pkl"):
    """Transform data and save scaler"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(save_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, save_path)
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    
    return data_scaled

def inverse_transform(data, load_path="Transforms/default/scaler.pkl"):
    """Inverse transform data using saved scaler"""
    
    # Convert to absolute path if it's relative
    if not os.path.isabs(load_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_path = os.path.join(base_dir, load_path)
    
    with open(load_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(data)

def split_data(X, use_val=True):
    if use_val:
        train_size = int(0.6 * len(X))
        val_size = int(0.2 * len(X))

        train = X[:train_size]
        val = X[train_size:train_size + val_size]
        test = X[train_size + val_size:]

        return train, val, test