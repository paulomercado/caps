import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import pandas as pd
import torch
from data import inverse_transform
import shap
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_residual_diagnostics(actual, pred, label, save_dir=None):
    actual    = np.array(actual).flatten()
    pred      = np.array(pred).flatten()
    residuals = actual - pred
    max_lags  = min(24, len(residuals) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Residual Diagnostics — {label}')

    # Residuals over time
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')

    # Distribution
    axes[0,1].hist(residuals, bins=20, density=True, alpha=0.7)
    xmin, xmax = axes[0,1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[0,1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r')
    axes[0,1].set_title('Distribution')

    # ACF / PACF
    plot_acf(residuals,  lags=max_lags, ax=axes[1,0], title='ACF')
    plot_pacf(residuals, lags=max_lags, ax=axes[1,1], title='PACF')

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/residual_diagnostics.png", dpi=150)
    plt.show()


def plot_qq(actual_dict, pred_dict, label_cols, save_path=None):
    fig, axes = plt.subplots(1, len(label_cols), figsize=(5 * len(label_cols), 4))
    if len(label_cols) == 1:
        axes = [axes]

    for ax, label in zip(axes, label_cols):
        residuals = np.array(actual_dict[label]).flatten() - np.array(pred_dict[label]).flatten()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot — {label}')
        ax.get_lines()[1].set_color('red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_test_predictions(test_preds, test_actuals, test_dates, label, save_dir=None):
    test_preds   = np.array(test_preds).flatten()
    test_actuals = np.array(test_actuals).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, test_actuals, label='Actual',    linewidth=2)
    plt.plot(test_dates, test_preds,   label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{label}: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    if save_dir:
        safe_name = label.replace(" ", "_").replace("/", "-").replace(":", "-")
        plt.savefig(f"{save_dir}/predictions_{safe_name}.png", dpi=150)
    plt.show()

def plot_all(actuals, dates, label, save_dir=None, covid_only=False, **model_preds):
    font = {'family': 'serif', 'color': 'k', 'weight': 'normal', 'size': 15}
    colors = [
        'red', 'green', 'blue', 'purple', 'orange',
        'teal', 'brown', 'magenta', 'olive', 'cyan',
        'coral', 'indigo', 'gold', 'darkgreen', 'navy'
    ]

    dates = pd.DatetimeIndex(dates)

    if covid_only:
        mask = (dates >= "2020-03-01") & (dates <= "2022-06-01")
        dates = dates[mask]
        actuals = np.array(actuals).flatten()[mask]
        model_preds = {k: np.array(v).flatten()[mask] for k, v in model_preds.items()}

    plt.figure(figsize=(14, 7))
    plt.plot(dates, actuals, label='Actual', color='k', linewidth=1.3)

    for i, (name, pred) in enumerate(model_preds.items()):
        plt.plot(dates, pred, label=name,
                 color=colors[i % len(colors)],
                 linewidth=0.9, alpha=0.75)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xlabel('Date', fontdict=font)
    plt.ylabel('PHP (Millions)', fontdict=font)
    plt.title(label, fontdict=font)
    plt.legend(loc='upper left', prop={'family': 'serif', 'size': 11},
               framealpha=0.9, edgecolor='gray')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe = label.replace(" ", "_").replace("/", "-").replace(":", "-")
        plt.savefig(os.path.join(save_dir, f'{safe}.png'), dpi=150)
    plt.show()

def explain_model(models, data_loader, args, num_samples=100):
    """Generate SHAP explanations for an ensemble of models."""
    for m in models:
        m.eval()

    # Build feature names
    feature_names = []
    if hasattr(args, 'features'):
        feature_names.extend(args.features)
    if hasattr(args, 'labels') and hasattr(args, 'lag_periods'):
        for label in args.labels:
            for lag in args.lag_periods:
                feature_names.append(f'{label}_lag_{lag}')
    if hasattr(args, 'dummy_vars'):
        feature_names.extend(args.dummy_vars)
    if getattr(args, 'use_seasonal', False):
        feature_names.extend(['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                              'is_tax_season', 'is_year_end'])

    # Collect background and test data
    background_data = []
    test_data = []
    for i, (inputs, _) in enumerate(data_loader):
        last_step = inputs[:, -1, :].cpu().numpy()
        if i == 0:
            background_data = last_step[:num_samples]
        test_data.append(last_step)
    
    test_data = np.vstack(test_data)
    n_test = min(50, len(test_data))
    test_data = test_data[:n_test]

    # Ensemble prediction wrapper
    def ensemble_predict(x):
        x_tensor = torch.FloatTensor(x).to(args.device).unsqueeze(1)
        preds = []
        with torch.no_grad():
            for m in models:
                preds.append(m(x_tensor).cpu().numpy())
        return np.mean(preds, axis=0)

    explainer = shap.KernelExplainer(ensemble_predict, background_data)
    shap_values = explainer.shap_values(test_data, nsamples=500,seed=1)

    return explainer, shap_values, test_data, feature_names


def plot_shap_summary(shap_values, test_data, feature_names, output_name=None, save_dir=None):
    """Plot SHAP summary - shows feature importance."""
    title = f'SHAP Feature Importance - {output_name}' if output_name else 'SHAP Feature Importance'
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_bar(shap_values, test_data, feature_names, output_name=None, save_dir=None):
    """Plot SHAP bar chart - mean absolute SHAP values."""
    title = f'SHAP Mean Importance - {output_name}' if output_name else 'SHAP Mean Importance'
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_waterfall(explainer, shap_values, test_data, feature_names, sample_idx=0, output_name=None, save_dir=None):
    """Plot SHAP waterfall - explains a single prediction."""
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=test_data[sample_idx],
        feature_names=feature_names,
    )
    title = f'SHAP Waterfall - Sample {sample_idx}, {output_name}' if output_name else f'SHAP Waterfall - Sample {sample_idx}'
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/shap_waterfall_sample{sample_idx}.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_shap_dependence(shap_values, test_data, feature_names, feature_idx, output_name=None, save_dir=None):
    """Plot SHAP dependence plot for a specific feature."""
    feature_name = feature_names[feature_idx]
    title = f'SHAP Dependence: {feature_name} - {output_name}' if output_name else f'SHAP Dependence: {feature_name}'
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, test_data, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/shap_dependence_{feature_name}.png", dpi=150, bbox_inches='tight')
    plt.show()
    
def plot_shap_combined(explainer, shap_values, test_data, feature_names, output_name=None, save_dir=None):
    """Save individual SHAP plots then stitch into one image."""
    import tempfile
    from PIL import Image
    tmp = tempfile.mkdtemp()
    
    # Summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False)
    plt.title(f'Feature Importance — {output_name}')
    plt.tight_layout()
    plt.savefig(f"{tmp}/summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'Mean Importance — {output_name}')
    plt.tight_layout()
    plt.savefig(f"{tmp}/bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Waterfall
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=test_data[0],
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f'Waterfall — Sample 0, {output_name}')
    plt.tight_layout()
    plt.savefig(f"{tmp}/waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Stitch horizontally
    imgs = [Image.open(f"{tmp}/{n}.png") for n in ['summary', 'bar', 'waterfall']]
    total_w = sum(i.width for i in imgs)
    max_h = max(i.height for i in imgs)
    combined = Image.new('RGB', (total_w, max_h), 'white')
    x = 0
    for img in imgs:
        combined.paste(img, (x, 0))
        x += img.width
    
    if save_dir:
        combined.save(f"{save_dir}/shap_combined.png", dpi=(150, 150))
    
    # Display in notebook
    plt.figure(figsize=(30, 8))
    plt.imshow(combined)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

