"""
SVM Model Training for Stock Price Prediction.

Builds and trains Support Vector Machine models (SVR for regression,
SVC for classification) for stock prediction. Uses scikit-learn.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
PLOTS_DIR = os.path.join(REPORTS_DIR, 'training_plots')
LOGS_DIR = os.path.join(REPORTS_DIR, 'training_logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_svr(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Train SVR with hyperparameter tuning via GridSearchCV.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional dict of hyperparameters to search

    Returns:
        model: Best SVR model
        metrics: dict of evaluation metrics
        y_pred: predictions on test set
    """
    if param_grid is None:
        param_grid = {
            'kernel': ['rbf'],
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.5],
        }

    tscv = TimeSeriesSplit(n_splits=3)
    svr = GridSearchCV(
        SVR(),
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )
    svr.fit(X_train, y_train)

    logger.info(f"SVR best params: {svr.best_params_}")

    y_pred = svr.predict(X_test)

    metrics = {
        'model': 'SVR',
        'task': 'Regression',
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'mse': round(mean_squared_error(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'r2': round(r2_score(y_test, y_pred), 4),
        'best_params': svr.best_params_,
    }

    logger.info(f"SVR Regression — MAE: {metrics['mae']}, RMSE: {metrics['rmse']}, R²: {metrics['r2']}")

    return svr.best_estimator_, metrics, y_pred


def train_svc(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Train SVC with hyperparameter tuning via GridSearchCV.

    Args:
        X_train, y_train: Training data (y = binary labels)
        X_test, y_test: Test data
        param_grid: Optional dict of hyperparameters

    Returns:
        model: Best SVC model
        metrics: dict of evaluation metrics
        y_pred: predictions on test set
    """
    if param_grid is None:
        param_grid = {
            'kernel': ['rbf'],
            'C': [1, 10, 100],
            'gamma': ['scale', 'auto'],
        }

    tscv = TimeSeriesSplit(n_splits=3)
    svc = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
    )
    svc.fit(X_train, y_train)

    logger.info(f"SVC best params: {svc.best_params_}")

    y_pred = svc.predict(X_test)

    metrics = {
        'model': 'SVC',
        'task': 'Classification',
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'best_params': svc.best_params_,
    }

    logger.info(f"SVC Classification — Acc: {metrics['accuracy']}, F1: {metrics['f1']}")

    return svc.best_estimator_, metrics, y_pred


def train_and_evaluate(data_path=None):
    """
    Full SVM training pipeline: load data, preprocess, train SVR+SVC, evaluate, save.

    Args:
        data_path: path to processed CSV

    Returns:
        dict with regression and classification metrics
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'final_stock_dataset_2026.csv')

    logger.info(f"Loading data from {data_path}")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        logger.warning("Data file not found. Fetching sample data...")
        try:
            from src.backend.data_fetcher import fetch_live_stock_data
            df = fetch_live_stock_data('AAPL', period='5y')
            df['Ticker'] = 'AAPL'
        except Exception as e:
            logger.error(f"Could not fetch data: {e}")
            return {'error': str(e)}

    # Use single ticker
    if 'Ticker' in df.columns:
        selected = df['Ticker'].value_counts().index[0]
        df = df[df['Ticker'] == selected].copy()
        logger.info(f"Training SVM on: {selected}")

    # Feature engineering
    try:
        from src.backend.feature_engineering import add_technical_features, add_targets, get_feature_columns
        df = add_technical_features(df)
        df = add_targets(df)
        feature_cols = get_feature_columns()
    except ImportError:
        logger.warning("Using fallback features")
        df['daily_return'] = df['Close'].pct_change()
        df['ma_7'] = df['Close'].rolling(7).mean()
        df['ma_30'] = df['Close'].rolling(30).mean()
        df['volatility'] = df['daily_return'].rolling(20).std()
        df['next_day_close'] = df['Close'].shift(-1)
        df['movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'daily_return', 'ma_7', 'ma_30', 'volatility']

    df = df.dropna().reset_index(drop=True)
    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Dataset: {len(df)} rows, {len(available)} features")

    # Train-test split (time-based)
    split = int(len(df) * 0.8)
    X_train = df[available].iloc[:split].values
    X_test = df[available].iloc[split:].values
    y_train_reg = df['next_day_close'].iloc[:split].values
    y_test_reg = df['next_day_close'].iloc[split:].values
    y_train_cls = df['movement'].iloc[:split].values
    y_test_cls = df['movement'].iloc[split:].values

    # Scale features (SVM requires scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale target for regression (SVR works better with scaled targets)
    target_scaler = StandardScaler()
    y_train_reg_scaled = target_scaler.fit_transform(y_train_reg.reshape(-1, 1)).flatten()
    y_test_reg_scaled = target_scaler.transform(y_test_reg.reshape(-1, 1)).flatten()

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Train SVR ────────────────────────────────────────────────────────────
    logger.info("Training SVR...")
    svr_model, reg_metrics, y_pred_reg_scaled = train_svr(
        X_train_scaled, y_train_reg_scaled, X_test_scaled, y_test_reg_scaled
    )

    # Inverse transform predictions for actual price metrics
    y_pred_reg_actual = target_scaler.inverse_transform(y_pred_reg_scaled.reshape(-1, 1)).flatten()
    reg_metrics['mae'] = round(mean_absolute_error(y_test_reg, y_pred_reg_actual), 4)
    reg_metrics['mse'] = round(mean_squared_error(y_test_reg, y_pred_reg_actual), 4)
    reg_metrics['rmse'] = round(np.sqrt(reg_metrics['mse']), 4)
    reg_metrics['r2'] = round(r2_score(y_test_reg, y_pred_reg_actual), 4)

    # ── Train SVC ────────────────────────────────────────────────────────────
    logger.info("Training SVC...")
    svc_model, cls_metrics, y_pred_cls = train_svc(
        X_train_scaled, y_train_cls, X_test_scaled, y_test_cls
    )

    # ── Save models ──────────────────────────────────────────────────────────
    import joblib
    joblib.dump(svr_model, os.path.join(MODEL_DIR, 'svm_regressor.pkl'))
    joblib.dump(svc_model, os.path.join(MODEL_DIR, 'svm_classifier.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'svm_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, 'svm_target_scaler.pkl'))
    logger.info("SVM models saved!")

    # ── Plot: Actual vs Predicted ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test_reg[:200], label='Actual', color='#818CF8', linewidth=1.5)
    ax.plot(y_pred_reg_actual[:200], label='Predicted', color='#F87171', linewidth=1.5, linestyle='--')
    ax.set_title('SVM: Actual vs Predicted Stock Prices', fontsize=14)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'svm_actual_vs_predicted.png'), dpi=150)
    plt.close()

    # ── Plot: Confusion Matrix ───────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test_cls, y_pred_cls)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Down', 'Up']).plot(ax=ax, cmap='Blues')
    ax.set_title('SVM Classifier — Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'svm_confusion_matrix.png'), dpi=150)
    plt.close()

    # ── Save training log ────────────────────────────────────────────────────
    log = {
        'timestamp': datetime.now().isoformat(),
        'regression': reg_metrics,
        'classification': cls_metrics,
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(LOGS_DIR, 'svm_training_log.json'), 'w') as f:
        json.dump(log, f, indent=2, default=convert)

    return {'regression': reg_metrics, 'classification': cls_metrics}


if __name__ == '__main__':
    print("=" * 60)
    print("SVM Stock Price Prediction - Training Pipeline")
    print("=" * 60)

    results = train_and_evaluate()

    if 'error' not in results:
        reg = results['regression']
        cls = results['classification']
        print(f"\nSVR Regression:  MAE={reg['mae']}, RMSE={reg['rmse']}, R²={reg['r2']}")
        print(f"SVC Classification: Acc={cls['accuracy']}, F1={cls['f1']}")
        print("\n" + "=" * 60)
        print("SVM Training Complete!")
        print("=" * 60)
    else:
        print(f"Training failed: {results['error']}")
