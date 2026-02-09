import pandas as pd
import numpy as np
import random  
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import GRUModel
from data import inverse_transform, transform_data, TimeSeriesDataset, split_data  

from sklearn.model_selection import TimeSeriesSplit

class Arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predictions, targets):       
        return torch.sqrt(self.mse(predictions, targets) + 1e-6)
        
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon  # Small constant to prevent division by zero

    def forward(self, output, target):
        return torch.mean(torch.abs((target - output) / (target + self.epsilon)))*100

def set_seed(seed=None):
    if seed is None:
        seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Remove CUDA-specific settings for Mac
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    os.environ['PYTHONHASHSEED'] = str(seed)

def load_dataset(args):
    """Load and prepare all datasets"""
    
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data")
    
    # Load raw data files with absolute paths
    btr_data = pd.read_csv(os.path.join(data_dir, "cordata.csv"))
    btr_data = btr_data.set_index("Date")

    macro_data = pd.read_csv(os.path.join(data_dir, "disaggregated.csv"))
    macro_data = macro_data.rename(columns={'Unnamed: 3': 'Date'}).set_index('Date')
    macro_data = macro_data[['TotalTrade_PHPMN', 'NominalGDP_disagg', 'Pop_disagg']]

    dummy = pd.read_csv(os.path.join(data_dir, "dummy.csv")).fillna(0)
    dummy = dummy.set_index("Date")

    btr_data.index = pd.to_datetime(btr_data.index)
    macro_data.index = pd.to_datetime(macro_data.index)
    dummy.index = pd.to_datetime(dummy.index)
    
    
    btr_data = btr_data.sort_index()
    macro_data = macro_data.sort_index()
    dummy = dummy.sort_index()

    # Filter by date e
    start = pd.to_datetime("1992-01-01")
    btr_data = btr_data[btr_data.index >= start]
    macro_data = macro_data[macro_data.index >= start]
    dummy = dummy[dummy.index >= start]

    # Join data
    df = btr_data.join(macro_data, how="inner").join(dummy, how="inner")

    # Create X and y
    X = pd.concat(
        [btr_data[['BIR', 'BOC', 'Other Offices']], macro_data, dummy],
        axis=1
    ).values.copy()
    y = df[['BIR', 'BOC', 'Other Offices']].values.copy()

    # Split into train/val/test
    train_data, val_data, test_data = split_data(X)
    train_labels, val_labels, test_labels = split_data(y)
    
    # Combine train + val for cross-validation
    cv_data = np.concatenate([train_data, val_data], axis=0)
    cv_labels = np.concatenate([train_labels, val_labels], axis=0)

    # Return everything needed
    return {
        'cv_data': cv_data,
        'cv_labels': cv_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'input_size': cv_data.shape[1],
        'output_size': cv_labels.shape[1]
    }

def train_model(model, dataloader, device, optimizer, criterion):

    model.train()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader, device, criterion, args):

    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)


        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        all_preds.append(outputs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, all_preds

def run(model, train_loader, val_loader, test_loader, args, fold=None):
    """
    Train and evaluate model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        args: Arguments object containing all hyperparameters
        fold: Optional fold number for cross-validation
    """
    
    set_seed(args.seed if hasattr(args, 'seed') else 1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=args.factor, 
        patience=args.patience
    )
    
    train_losses = []
    val_losses = []

    for e in range(args.epoch):
        train_loss = train_model(model, train_loader, args.device, optimizer, args.train_criterion)
        val_loss, _ = evaluate(model, val_loader, args.device, args.train_criterion, args)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (e + 1) % 100 == 0 or e == 0:
            fold_prefix = f"Fold {fold + 1} - " if fold is not None else ""
            print(f"{fold_prefix}Epoch {e+1}/{args.epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    _, test_preds = evaluate(model, test_loader, args.device, args.test_criterion, args)

    # Inverse transform using fold-specific or default scaler
    if fold is not None:
        scaler_name = f"Transforms/labels_scaled{fold}"
    elif args.final is True:
        scaler_name = f"Transforms/final_labels_scaled"
    else:
        scaler_name = f"Transforms/labels_scaled"
    
    inversed_test_preds = inverse_transform(test_preds, scaler_name)
    
    # Get corresponding actual labels from test_loader 
    actual_labels = []
    for _, targets in test_loader:
        actual_labels.append(targets)
    actual_labels = torch.cat(actual_labels, dim=0).cpu().numpy()
    
    # Inverse transform actual labels 
    inversed_actual = inverse_transform(actual_labels, scaler_name)
    
    # Calculate test loss
    test_loss = args.test_criterion(
        torch.tensor(inversed_test_preds), 
        torch.tensor(inversed_actual)
    )
    
    return test_loss, inversed_test_preds, train_losses, val_losses

def crossval(data, labels, test_data, test_labels, args, n_splits=5):

    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Split data for this fold
        train_data_fold = data[train_idx]
        train_labels_fold = labels[train_idx]
        val_data_fold = data[val_idx]
        val_labels_fold = labels[val_idx]
        
        # Scale data
        set_seed(1)
        train_data_scaled = transform_data(train_data_fold, f"Transforms/train_scaled_{fold}")
        train_labels_scaled = transform_data(train_labels_fold, f"Transforms/labels_scaled{fold}")
        val_data_scaled = transform_data(val_data_fold, f"Transforms/train_scaled_{fold}")
        val_labels_scaled = transform_data(val_labels_fold, f"Transforms/labels_scaled{fold}")
        test_data_scaled = transform_data(test_data, f"Transforms/train_scaled_{fold}")
        test_labels_scaled = transform_data(test_labels, f"Transforms/labels_scaled{fold}")
        
        input_size = train_data_scaled.shape[1]
        output_size = train_labels_scaled.shape[1]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data_scaled, train_labels_scaled, seq_len=args.seq_len)
        val_dataset = TimeSeriesDataset(val_data_scaled, val_labels_scaled, seq_len=args.seq_len)
        test_dataset = TimeSeriesDataset(test_data_scaled, test_labels_scaled, seq_len=args.seq_len)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create fresh model for this fold
        set_seed(1)
        model = GRUModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(args.device)
        
        # Pass fold number to run function
        test_loss, test_preds, train_losses, val_losses = run(
            model, 
            train_loader, 
            val_loader, 
            test_loader, 
            args,
            fold=fold  # Pass fold number
        )
        
        # Store results
        fold_results.append({
            'fold': fold + 1,
            'test_loss': test_loss.item() if torch.is_tensor(test_loss) else test_loss,
            'test_preds': test_preds,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })
        
        print(f"Fold {fold + 1} Test Loss: {test_loss:.4f}")
    
    # Summary statistics
    test_losses = [r['test_loss'] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results:")
    print(f"{'='*50}")
    print(f"Mean Test Loss: {np.mean(test_losses):.4f} (+/- {np.std(test_losses):.4f})")
    print(f"Min Test Loss: {np.min(test_losses):.4f}")
    print(f"Max Test Loss: {np.max(test_losses):.4f}")
    
    return fold_results