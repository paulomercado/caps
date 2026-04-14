import torch
import torch.nn as nn
import numpy as np
from ray import tune  

def ray_train(config):
    from train import (
        load_dataset, crossval, set_seed, Arguments, MAPELoss, RMSELoss, MADLoss, QuantileLoss
    )
    import torch.nn as nn

    data_args = Arguments(
        features=config.get('features'),
        labels=config.get('labels'),
        dummy_vars=config.get('dummy_vars'),
        experiment_name=config.get('experiment_name', 'default'),
        lag_periods=config.get('lag_periods', [1, 3, 6, 12]),
        use_seasonal=config.get('use_seasonal', False),
        use_lags=config.get('use_lags', True),
        start_date=config.get('start_date', '2000-01-01')
    )

    dataset = load_dataset(data_args)

    loss_dict = {
        "huber":    nn.HuberLoss(),
        "mse":      nn.MSELoss(),
        "mae":      nn.L1Loss(),
        "quantile": QuantileLoss(),
        "madl":     MADLoss(),
        "rmse":     RMSELoss()
    }
    train_criterion = loss_dict[config.get('train_loss_name', 'mse')]

    args = Arguments(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        lr=config['lr'],
        wd=config['wd'],
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        l1_lambda=config['l1_lambda'],
        train_loss_name=config.get('train_loss_name', 'mse'),
        forecast_horizon=config.get('forecast_horizon', 1),
        features=config.get('features'),
        labels=config.get('labels'),
        dummy_vars=config.get('dummy_vars'),
        lag_periods=config.get('lag_periods', [1, 3, 6, 12]),
        use_seasonal=config.get('use_seasonal', False),
        use_lags=config.get('use_lags', True),
        experiment_name=config.get('experiment_name', 'default'),
        n_splits=config.get('n_splits', 5),
        seed=1,
        epoch=200,
        tuning_mode=True,
        cv_data=dataset['cv_data'],
        cv_labels=dataset['cv_labels'],
        test_data=dataset['test_data'],
        test_labels=dataset['test_labels'],
        input_size=dataset['input_size'],
        output_size=dataset['output_size'],
        early_stopping_patience=config.get('early_stopping_patience', 75),
        device=torch.device("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu"),
        train_criterion=train_criterion,
        test_criterion=MAPELoss(),
    )
    set_seed(args.seed)

    fold_results = crossval(data=args.cv_data, labels=args.cv_labels, args=args)

    mean_loss = float(np.mean([r['test_loss'] for r in fold_results]))
    std_loss  = float(np.std([r['test_loss']  for r in fold_results]))

    all_label_keys  = fold_results[0]['per_label_mape'].keys()
    per_label_means = {k: float(np.mean([r['per_label_mape'][k] for r in fold_results]))
                       for k in all_label_keys}
    mean_peak = float(np.mean([r['peak_mape'] for r in fold_results]))
    mean_dir  = float(np.mean([r['dir_acc']   for r in fold_results]))
    mean_combined = float(np.mean([r['combined'] for r in fold_results]))
    tune.report({
        "loss":      mean_loss,
        "std":       std_loss,
        "peak_loss": mean_peak,
        "dir_acc":   mean_dir,
        "combined":  mean_combined,
        **per_label_means
    })