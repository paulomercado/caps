# tuning.py
import torch
import torch.nn as nn
import numpy as np
from ray import tune  # Change this line

def ray_train(config):
    """Self-contained Ray Tune training function"""
    
    # Import all your custom functions
    from train import (
        load_dataset,
        crossval,
        set_seed,
        Arguments,
        MAPELoss
    )
    
    # Load dataset
    dataset = load_dataset(None)
    
    # Create args
    args = Arguments(
        **config,
        seed=1,
        epoch=200,
        
        cv_data=dataset['cv_data'],
        cv_labels=dataset['cv_labels'],
        test_data=dataset['test_data'],
        test_labels=dataset['test_labels'],
        input_size=dataset['input_size'],
        output_size=dataset['output_size'],
        
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        train_criterion=nn.MSELoss(),
        test_criterion=MAPELoss(),
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Run cross-validation
    fold_results = crossval(
        data=args.cv_data,
        labels=args.cv_labels,
        test_data=args.cv_data,
        test_labels=args.cv_labels,
        args=args,
        n_splits=5
    )
    
    # Report results
    mean_loss = np.mean([r['test_loss'] for r in fold_results])
    std_loss = np.std([r['test_loss'] for r in fold_results])
    
    tune.report({"loss": mean_loss, "std": std_loss})  # Change from session.report to tune.report