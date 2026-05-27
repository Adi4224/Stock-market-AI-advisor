"""
Train Random Forest models for stock price prediction and movement classification.

This module provides functions to train, tune, and evaluate Random Forest models
for both regression (next-day close price prediction) and classification (up/down
movement prediction) tasks using GridSearchCV with TimeSeriesSplit cross-validation.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'daily_return', 'ma_7', 'ma_30', 'ma_100',
    'volatility', 'rsi', 'macd', 'macd_signal', 'volume_change'
]
REGRESSION_TARGET = 'next_day_close'
CLASSIFICATION_TARGET = 'movement'


# ===================================================================
# Training functions
# ===================================================================

def train_rf_regressor(X_train, y_train, params=None):
    """Train a Random Forest Regressor with GridSearchCV hyper-parameter tuning.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training target values (next-day close price).
    params : dict or None, optional
        Custom parameter grid for GridSearchCV.  If *None*, a sensible default
        grid is used.

    Returns
    -------
    tuple
        (best_estimator, best_params) – the fitted model and the best
        hyper-parameter combination found.
    """
    if params is None:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
        }

    logger.info("Starting Random Forest Regressor training with GridSearchCV …")
    logger.info("Parameter grid: %s", params)

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best regressor params: %s", grid_search.best_params_)
    logger.info("Best CV score (neg MSE): %.4f", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_


def train_rf_classifier(X_train, y_train, params=None):
    """Train a Random Forest Classifier with GridSearchCV hyper-parameter tuning.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples,)
        Training binary labels (1 = Up, 0 = Down).
    params : dict or None, optional
        Custom parameter grid for GridSearchCV.

    Returns
    -------
    tuple
        (best_estimator, best_params)
    """
    if params is None:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
        }

    logger.info("Starting Random Forest Classifier training with GridSearchCV …")
    logger.info("Parameter grid: %s", params)

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    logger.info("Best classifier params: %s", grid_search.best_params_)
    logger.info("Best CV accuracy: %.4f", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_


# ===================================================================
# Plotting helpers
# ===================================================================

def _plot_feature_importance(model, feature_names, title, save_path):
    """Plot and save a horizontal bar chart of feature importances.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must expose ``feature_importances_``.
    feature_names : list[str]
        Names corresponding to each feature.
    title : str
        Plot title.
    save_path : str
        File path to save the figure.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)


# ===================================================================
# Main pipeline
# ===================================================================

def train_and_evaluate(data_path=None):
    """End-to-end training and evaluation pipeline for Random Forest models.

    1. Loads processed data.
    2. Performs a time-based 80/20 train/test split.
    3. Trains RF regressor and classifier with GridSearchCV.
    4. Evaluates on the test set.
    5. Saves models to ``models/`` and plots to ``reports/training_plots/``.

    Parameters
    ----------
    data_path : str or None
        Path to the processed CSV file.  Defaults to the project's standard
        data location.

    Returns
    -------
    dict
        Dictionary containing regression and classification evaluation metrics
        along with the best hyper-parameters for each model.
    """
    if data_path is None:
        data_path = DATA_PATH

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s", data_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise

    logger.info("Dataset shape: %s", df.shape)

    # Drop rows with NaN in features or targets
    required_cols = FEATURE_COLS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=required_cols)
    logger.info("Dataset shape after dropping NaNs: %s", df.shape)

    X = df[FEATURE_COLS].values
    y_reg = df[REGRESSION_TARGET].values
    y_cls = df[CLASSIFICATION_TARGET].values

    # ------------------------------------------------------------------
    # 2. Time-based split (80 / 20)
    # ------------------------------------------------------------------
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    reg_model, reg_params = train_rf_regressor(X_train, y_reg_train)
    cls_model, cls_params = train_rf_classifier(X_train, y_cls_train)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    y_reg_pred = reg_model.predict(X_test)
    y_cls_pred = cls_model.predict(X_test)

    reg_metrics = {
        'model': 'RandomForest_Regressor',
        'mae': float(mean_absolute_error(y_reg_test, y_reg_pred)),
        'mse': float(mean_squared_error(y_reg_test, y_reg_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
        'r2': float(r2_score(y_reg_test, y_reg_pred)),
        'best_params': reg_params,
    }
    cls_metrics = {
        'model': 'RandomForest_Classifier',
        'accuracy': float(accuracy_score(y_cls_test, y_cls_pred)),
        'precision': float(precision_score(y_cls_test, y_cls_pred, zero_division=0)),
        'recall': float(recall_score(y_cls_test, y_cls_pred, zero_division=0)),
        'f1': float(f1_score(y_cls_test, y_cls_pred, zero_division=0)),
        'best_params': cls_params,
    }

    logger.info("Regression metrics: %s", reg_metrics)
    logger.info("Classification metrics: %s", cls_metrics)

    # ------------------------------------------------------------------
    # 5. Save models
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    reg_path = os.path.join(MODELS_DIR, 'random_forest_regressor.pkl')
    cls_path = os.path.join(MODELS_DIR, 'random_forest_classifier.pkl')
    joblib.dump(reg_model, reg_path)
    joblib.dump(cls_model, cls_path)
    logger.info("Regressor saved to %s", reg_path)
    logger.info("Classifier saved to %s", cls_path)

    # ------------------------------------------------------------------
    # 6. Feature importance plot
    # ------------------------------------------------------------------
    _plot_feature_importance(
        reg_model,
        FEATURE_COLS,
        'Random Forest Regressor – Feature Importance',
        os.path.join(REPORTS_DIR, 'rf_feature_importance.png'),
    )

    results = {
        'regression': reg_metrics,
        'classification': cls_metrics,
    }
    return results


# ===================================================================
# CLI entry-point
# ===================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Random Forest Training Pipeline")
    logger.info("=" * 60)

    try:
        results = train_and_evaluate()
        print("\n===== Random Forest Results =====")
        print("\nRegression Metrics:")
        for k, v in results['regression'].items():
            print(f"  {k}: {v}")
        print("\nClassification Metrics:")
        for k, v in results['classification'].items():
            print(f"  {k}: {v}")
    except Exception as e:
        logger.error("Training pipeline failed: %s", e, exc_info=True)
        raise
