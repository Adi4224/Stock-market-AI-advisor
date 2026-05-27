"""
LSTM Model Training for Stock Price Prediction.

Builds and trains a Long Short-Term Memory (LSTM) neural network for
time-series stock price prediction. TensorFlow/Keras is a required dependency.

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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def create_sequences(data, target, lookback=60):
    """
    Create sliding window sequences for LSTM input.

    Args:
        data: numpy array of features, shape (n_samples, n_features)
        target: numpy array of target values, shape (n_samples,)
        lookback: number of past time steps to use as input

    Returns:
        X: numpy array of shape (n_sequences, lookback, n_features)
        y: numpy array of shape (n_sequences,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Build a stacked LSTM model for stock price prediction.

    Args:
        input_shape: tuple (lookback, n_features)
        units: number of LSTM units per layer
        dropout_rate: dropout rate between layers
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    logger.info(f"LSTM model built: input_shape={input_shape}, units={units}")
    model.summary(print_fn=logger.info)

    return model


def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the LSTM model with early stopping and learning rate reduction.

    Args:
        X_train: training features (n_samples, lookback, n_features)
        y_train: training targets
        X_val: validation features
        y_val: validation targets
        epochs: maximum number of training epochs
        batch_size: training batch size

    Returns:
        model: trained Keras model
        history: training history object
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    logger.info(f"Training LSTM: epochs={epochs}, batch_size={batch_size}")
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def plot_training_history(history, save_path=None):
    """Plot and optionally save training history (loss and MAE curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', color='#00D4FF')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#FF5252')
    axes[0].set_title('LSTM Training & Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE', color='#00C853')
    axes[1].plot(history.history['val_mae'], label='Validation MAE', color='#FFD700')
    axes[1].set_title('LSTM Training & Validation MAE', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    plt.close()


def plot_actual_vs_predicted(y_actual, y_predicted, save_path=None):
    """Plot actual vs predicted stock prices."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(y_actual[:200], label='Actual Price', color='#00D4FF', linewidth=1.5)
    ax.plot(y_predicted[:200], label='Predicted Price', color='#FF5252', linewidth=1.5, linestyle='--')
    ax.set_title('LSTM: Actual vs Predicted Stock Prices', fontsize=14)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Actual vs Predicted plot saved to {save_path}")

    plt.close()


def train_and_evaluate(data_path=None, lookback=60, epochs=100, batch_size=32):
    """
    Full training pipeline: load data, preprocess, train LSTM, evaluate, save.

    Args:
        data_path: path to the processed stock dataset CSV
        lookback: number of past days to use for prediction
        epochs: max training epochs
        batch_size: training batch size

    Returns:
        dict with evaluation metrics and training info
    """
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'final_stock_dataset_2026.csv')

    # Load data
    logger.info(f"Loading data from {data_path}")

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        logger.warning(f"Data file not found at {data_path}. Fetching sample data...")
        try:
            from src.backend.data_fetcher import fetch_live_stock_data
            df = fetch_live_stock_data('AAPL', period='5y')
            df['Ticker'] = 'AAPL'
        except Exception as e:
            logger.error(f"Could not fetch sample data: {e}")
            return {'error': str(e)}

    # Use single ticker for LSTM (works best on individual stocks)
    if 'Ticker' in df.columns:
        tickers = df['Ticker'].unique()
        selected_ticker = tickers[0]
        df = df[df['Ticker'] == selected_ticker].copy()
        logger.info(f"Training LSTM on ticker: {selected_ticker}")

    # Add technical features
    try:
        from src.backend.feature_engineering import add_technical_features, get_feature_columns
        df = add_technical_features(df)
        feature_cols = get_feature_columns()
    except ImportError:
        # Fallback feature engineering
        logger.warning("Using fallback feature engineering")
        df['daily_return'] = df['Close'].pct_change()
        df['ma_7'] = df['Close'].rolling(7).mean()
        df['ma_30'] = df['Close'].rolling(30).mean()
        df['ma_100'] = df['Close'].rolling(100).mean()
        df['volatility'] = df['daily_return'].rolling(20).std()
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
                        'ma_7', 'ma_30', 'ma_100', 'volatility']

    # Clean data
    df = df.dropna().reset_index(drop=True)

    if len(df) < lookback + 100:
        logger.error(f"Not enough data: {len(df)} rows (need at least {lookback + 100})")
        return {'error': 'Insufficient data for LSTM training'}

    # Prepare features and target
    available_features = [col for col in feature_cols if col in df.columns]
    features = df[available_features].values
    target = df['Close'].values

    logger.info(f"Dataset: {len(df)} rows, {len(available_features)} features")

    # Scale data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(features_scaled, target_scaled, lookback=lookback)
    logger.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")

    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")

    # Train model
    model, history = train_lstm_model(X_train, y_train, X_val, y_val,
                                       epochs=epochs, batch_size=batch_size)

    # Predict
    y_pred_scaled = model.predict(X_val, verbose=0).flatten()

    # Inverse transform predictions
    y_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_predicted = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Evaluate
    mae = mean_absolute_error(y_actual, y_predicted)
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_predicted)

    metrics = {
        'model': 'LSTM',
        'task': 'Regression',
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': round(history.history['loss'][-1], 6),
        'final_val_loss': round(history.history['val_loss'][-1], 6),
        'lookback': lookback,
        'features_used': available_features,
        'training_samples': X_train.shape[0],
        'validation_samples': X_val.shape[0]
    }

    logger.info(f"LSTM Results: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
    model.save(model_path)
    logger.info(f"LSTM model saved to {model_path}")

    # Save scalers
    feature_scaler_path = os.path.join(MODEL_DIR, 'lstm_feature_scaler.pkl')
    target_scaler_path = os.path.join(MODEL_DIR, 'lstm_target_scaler.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.pkl')

    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, f)

    logger.info("LSTM scalers saved")

    # Save plots
    plot_training_history(
        history,
        save_path=os.path.join(PLOTS_DIR, 'lstm_training_history.png')
    )
    plot_actual_vs_predicted(
        y_actual, y_predicted,
        save_path=os.path.join(PLOTS_DIR, 'lstm_actual_vs_predicted.png')
    )

    # Save training log
    log_path = os.path.join(LOGS_DIR, 'lstm_training_log.json')
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'training_history': {
            'loss': [round(v, 6) for v in history.history['loss']],
            'val_loss': [round(v, 6) for v in history.history['val_loss']],
            'mae': [round(v, 6) for v in history.history['mae']],
            'val_mae': [round(v, 6) for v in history.history['val_mae']]
        }
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Training log saved to {log_path}")

    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("LSTM Stock Price Prediction - Training Pipeline")
    if not HAS_TENSORFLOW:
        print("ERROR: TensorFlow is not installed!")
        print("Install TensorFlow first: pip install tensorflow")
        print("Note: TensorFlow requires Python <= 3.12")
        sys.exit(1)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 60)

    results = train_and_evaluate(lookback=60, epochs=100, batch_size=32)

    if 'error' not in results:
        print("\n" + "=" * 60)
        print("LSTM Training Complete!")
        print(f"  MAE:  {results['mae']}")
        print(f"  MSE:  {results['mse']}")
        print(f"  RMSE: {results['rmse']}")
        print(f"  R²:   {results['r2']}")
        print(f"  Epochs trained: {results['epochs_trained']}")
        print("=" * 60)
    else:
        print(f"\nTraining failed: {results['error']}")
