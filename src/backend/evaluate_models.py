"""
Model Evaluation Utilities.

Provides functions to evaluate regression and classification models,
generate comparison tables, and create evaluation visualizations.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
PLOTS_DIR = os.path.join(REPORTS_DIR, 'training_plots')
LOGS_DIR = os.path.join(REPORTS_DIR, 'training_logs')

for d in [REPORTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


def evaluate_regression(y_true, y_pred, model_name='Model'):
    """
    Evaluate a regression model.

    Args:
        y_true: actual target values
        y_pred: predicted values
        model_name: name of the model for reporting

    Returns:
        dict with MAE, MSE, RMSE, R² score
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'model': model_name,
        'task': 'Regression',
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'r2': round(float(r2), 4)
    }


def evaluate_classification(y_true, y_pred, model_name='Model'):
    """
    Evaluate a classification model.

    Args:
        y_true: actual labels
        y_pred: predicted labels
        model_name: name of the model for reporting

    Returns:
        dict with accuracy, precision, recall, F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'model': model_name,
        'task': 'Classification',
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4)
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Create and optionally save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def plot_actual_vs_predicted(y_true, y_pred, model_name, save_path=None):
    """Create actual vs predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, color='#00D4FF')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_title(f'{model_name}: Actual vs Predicted', fontsize=14)
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].grid(True, alpha=0.3)

    # Time series comparison (first 200 points)
    n = min(200, len(y_true))
    axes[1].plot(range(n), y_true[:n], label='Actual', color='#00D4FF', linewidth=1.5)
    axes[1].plot(range(n), y_pred[:n], label='Predicted', color='#FF5252',
                 linewidth=1.5, linestyle='--')
    axes[1].set_title(f'{model_name}: Price Comparison', fontsize=14)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Create residual plot for regression evaluation."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10, color='#00C853')
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_title(f'{model_name}: Residual Plot', fontsize=14)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=50, color='#FFD700', alpha=0.7, edgecolor='white')
    axes[1].set_title(f'{model_name}: Residual Distribution', fontsize=14)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_table(results):
    """
    Create a model comparison DataFrame from evaluation results.

    Args:
        results: list of result dicts from evaluate_regression/evaluate_classification

    Returns:
        pd.DataFrame with comparison metrics
    """
    df = pd.DataFrame(results)

    # Save to reports
    csv_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)

    return df


def generate_full_report(results, save_dir=None):
    """
    Generate complete evaluation report with all plots and comparison table.

    Args:
        results: list of evaluation result dicts
        save_dir: directory to save report files
    """
    if save_dir is None:
        save_dir = REPORTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Create comparison bar charts
    regression_results = [r for r in results if r.get('task') == 'Regression']
    classification_results = [r for r in results if r.get('task') == 'Classification']

    if regression_results:
        _plot_regression_comparison(regression_results,
                                    os.path.join(PLOTS_DIR, 'regression_comparison.png'))

    if classification_results:
        _plot_classification_comparison(classification_results,
                                        os.path.join(PLOTS_DIR, 'classification_comparison.png'))

    # Save text report
    report_path = os.path.join(save_dir, 'final_results.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STOCK MARKET AI ADVISOR - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("REGRESSION MODELS:\n")
        f.write("-" * 40 + "\n")
        for r in regression_results:
            f.write(f"\n{r['model']}:\n")
            f.write(f"  MAE:  {r.get('mae', 'N/A')}\n")
            f.write(f"  MSE:  {r.get('mse', 'N/A')}\n")
            f.write(f"  RMSE: {r.get('rmse', 'N/A')}\n")
            f.write(f"  R²:   {r.get('r2', 'N/A')}\n")

        f.write(f"\n\nCLASSIFICATION MODELS:\n")
        f.write("-" * 40 + "\n")
        for r in classification_results:
            f.write(f"\n{r['model']}:\n")
            f.write(f"  Accuracy:  {r.get('accuracy', 'N/A')}\n")
            f.write(f"  Precision: {r.get('precision', 'N/A')}\n")
            f.write(f"  Recall:    {r.get('recall', 'N/A')}\n")
            f.write(f"  F1 Score:  {r.get('f1', 'N/A')}\n")

        # Best model selection
        if regression_results:
            best_reg = min(regression_results, key=lambda x: x.get('rmse', float('inf')))
            f.write(f"\n\nBEST REGRESSION MODEL: {best_reg['model']}\n")
            f.write(f"  Reason: Lowest RMSE ({best_reg.get('rmse', 'N/A')})\n")

        if classification_results:
            best_cls = max(classification_results, key=lambda x: x.get('f1', 0))
            f.write(f"\nBEST CLASSIFICATION MODEL: {best_cls['model']}\n")
            f.write(f"  Reason: Highest F1 Score ({best_cls.get('f1', 'N/A')})\n")

        f.write(f"\n\n{'=' * 60}\n")
        f.write("DISCLAIMER: This project is for educational purposes only.\n")
        f.write("It does not provide financial advice.\n")
        f.write("=" * 60 + "\n")

    return comparison_df


def _plot_regression_comparison(results, save_path):
    """Create regression metrics comparison bar chart."""
    models = [r['model'] for r in results]
    metrics = ['mae', 'rmse', 'r2']
    colors = ['#00D4FF', '#FF5252', '#00C853']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results]
        bars = axes[i].bar(models, values, color=colors[i], alpha=0.8, edgecolor='white')
        axes[i].set_title(metric.upper(), fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Regression Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_classification_comparison(results, save_path):
    """Create classification metrics comparison bar chart."""
    models = [r['model'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#00D4FF', '#FFD700', '#FF5252', '#00C853']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [r.get(metric, 0) for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                      color=color, alpha=0.8, edgecolor='white')

    ax.set_title('Classification Model Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Demo evaluation with random data
    np.random.seed(42)
    y_true_reg = np.random.uniform(100, 200, 100)
    y_pred_reg = y_true_reg + np.random.normal(0, 5, 100)

    result = evaluate_regression(y_true_reg, y_pred_reg, 'Demo Model')
    print("Regression Evaluation:", result)

    y_true_cls = np.random.randint(0, 2, 100)
    y_pred_cls = np.random.randint(0, 2, 100)

    result_cls = evaluate_classification(y_true_cls, y_pred_cls, 'Demo Classifier')
    print("Classification Evaluation:", result_cls)
