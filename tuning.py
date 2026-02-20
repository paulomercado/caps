import torch
import torch.nn as nn
import numpy as np
from ray import tune  

def ray_train(config):
    from train import (
        load_dataset, crossval, set_seed, Arguments, MAPELoss, RMSELoss, DirectionalLoss
    )
    import torch.nn as nn

    # Only pass data-shape args to load_dataset
    data_args = Arguments(
        features=config.get('features'),
        labels=config.get('labels'),
        dummy_vars=config.get('dummy_vars'),
        experiment_name=config.get('experiment_name', 'default'),
        lag_periods=config.get('lag_periods', [1, 3, 6, 12]),
        use_branches=config.get('use_branches', True),
        use_attention=config.get('use_attention', False),
        use_seasonal=config.get('use_seasonal', False),
        use_lags=config.get('use_lags', True),
    )

    dataset = load_dataset(data_args)

    loss_dict = {
        "huber":       nn.HuberLoss(),
        "mse":         nn.MSELoss(),
        "mae":         nn.L1Loss(),
        "directional": DirectionalLoss(alpha=0.5),
        "rmse":        RMSELoss(),
    }
    train_criterion = loss_dict[config.get('train_loss_name', 'mse')]

    # Only architecture/optimizer keys from config — no data keys
    args = Arguments(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention=config['use_attention'],
        use_branches=config['use_branches'],
        use_se=config.get('use_se', False),
        lr=config['lr'],
        wd=config['wd'],
        factor=config['factor'],
        patience=config['patience'],
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        l1_lambda=config['l1_lambda'],
        train_loss_name=config.get('train_loss_name', 'mse'),
        # data
        features=config.get('features'),
        labels=config.get('labels'),
        dummy_vars=config.get('dummy_vars'),
        lag_periods=config.get('lag_periods', [1, 3, 6, 12]),
        use_seasonal=config.get('use_seasonal', False),
        use_lags=config.get('use_lags', True),
        experiment_name=config.get('experiment_name', 'default'),
        # training
        seed=1,
        epoch=100,
        tuning_mode=True,
        cv_data=dataset['cv_data'],
        cv_labels=dataset['cv_labels'],
        test_data=dataset['test_data'],
        test_labels=dataset['test_labels'],
        input_size=dataset['input_size'],
        output_size=dataset['output_size'],
        device=torch.device("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu"),
        train_criterion=train_criterion,
        test_criterion=MAPELoss(),
    )

    set_seed(args.seed)

    fold_results = crossval(data=args.cv_data, labels=args.cv_labels, args=args, n_splits=5)

    mean_loss = float(np.mean([r['test_loss'] for r in fold_results]))
    std_loss  = float(np.std([r['test_loss']  for r in fold_results]))

    all_label_keys  = fold_results[0]['per_label_mape'].keys()
    per_label_means = {k: float(np.mean([r['per_label_mape'][k] for r in fold_results]))
                       for k in all_label_keys}

    tune.report({"loss": mean_loss, "std": std_loss, **per_label_means})