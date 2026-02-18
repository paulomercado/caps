import torch
import torch.nn as nn
import numpy as np
from ray import tune  

def ray_train(config):

    
    # Import all your custom functions
    from train import (
        load_dataset,
        crossval,
        set_seed,
        Arguments,
        MAPELoss
    )

    data_args = Arguments(
        features=config.get('features', ['BIR', 'BOC', 'Other Offices', "Non-tax Revenues", "Expenditures", 'TotalTrade_PHPMN', 'NominalGDP_disagg', 'Pop_disagg']),
        labels=config.get('labels', ['BIR', 'BOC', 'Other Offices', "Non-tax Revenues", "Expenditures"]),
        dummy_vars=config.get('dummy_vars', ['COVID-19', 'TRAIN', 'CREATE', 'FIST', 'BIR_COMM']),
        experiment_name=config.get('experiment_name', 'default'),
        lag_periods=config.get('lag_periods', [1, 3, 12]),
        use_branches=config.get('use_branches', True),
        use_attention=config.get('use_attention', True),
        use_seasonal=config.get('use_seasonal', False),   # ← missing from your current version
    )
    # Load dataset
    dataset = load_dataset(data_args)
    
    # Create args
    args = Arguments(
        **config,
        seed=1,
        epoch=100,
        tuning_mode=True,       
        cv_data=dataset['cv_data'],
        cv_labels=dataset['cv_labels'],
        test_data=dataset['test_data'],
        test_labels=dataset['test_labels'],
        input_size=dataset['input_size'],
        output_size=dataset['output_size'],
        
        device = torch.device("mps" if torch.backends.mps.is_available() 
                              else "cuda" if torch.cuda.is_available()
                              else "cpu"),
        train_criterion=nn.HuberLoss(),
        test_criterion=MAPELoss()
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Run cross-validation
    fold_results = crossval(
        data=args.cv_data,
        labels=args.cv_labels,
        args=args,
        n_splits=5
    )
    mean_loss = np.mean([r['test_loss'] for r in fold_results])
    std_loss = np.std([r['test_loss'] for r in fold_results])
    # Report results
    all_label_keys = fold_results[0]['per_label_mape'].keys()
    per_label_means = {
        k: float(np.mean([r['per_label_mape'][k] for r in fold_results]))
        for k in all_label_keys
    }

    tune.report({"loss": mean_loss, "std": std_loss, **per_label_means})  # Change from session.report to tune.report