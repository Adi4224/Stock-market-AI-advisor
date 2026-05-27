# %% [markdown]
# # 06 — LSTM Model Training
# Train Long Short-Term Memory model for time-series stock price prediction.
# TensorFlow/Keras is REQUIRED.

# %%
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, get_feature_columns
from src.backend.train_lstm import create_sequences, build_lstm_model, plot_training_history, plot_actual_vs_predicted

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]: os.makedirs(d, exist_ok=True)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    from src.backend.data_fetcher import fetch_live_stock_data
    df = fetch_live_stock_data('AAPL', '5y').reset_index()
    df['Ticker'] = 'AAPL'

if 'Ticker' in df.columns:
    ticker = df['Ticker'].value_counts().index[0]
    df = df[df['Ticker'] == ticker].copy()
    print(f"Training on: {ticker}")

df = add_technical_features(df)
df = df.dropna().reset_index(drop=True)
feature_cols = [c for c in get_feature_columns() if c in df.columns]
print(f"Shape: {df.shape}, Features: {len(feature_cols)}")

# %% [markdown]
# ## 2. Scale Data

# %%
LOOKBACK = 60
features = df[feature_cols].values
target = df['Close'].values

feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

X, y = create_sequences(features_scaled, target_scaled, lookback=LOOKBACK)
print(f"Sequences: X={X.shape}, y={y.shape}")

# %% [markdown]
# ## 3. Train-Validation Split (Time-Based)

# %%
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

# %% [markdown]
# ## 4. Build and Train LSTM

# %%
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=64)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32, callbacks=callbacks, verbose=1)

# %% [markdown]
# ## 5. Plot Training History

# %%
plot_training_history(history, save_path=os.path.join(PLOTS_DIR, 'lstm_training_history.png'))

# %% [markdown]
# ## 6. Evaluate

# %%
y_pred_scaled = model.predict(X_val, verbose=0).flatten()
y_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_predicted = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_predicted)

print(f"\nLSTM Results:")
print(f"  MAE:  {mae:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

plot_actual_vs_predicted(y_actual, y_predicted, save_path=os.path.join(PLOTS_DIR, 'lstm_actual_vs_predicted.png'))

# %% [markdown]
# ## 7. Save Model and Scalers

# %%
import pickle

model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))
with open(os.path.join(MODEL_DIR, 'lstm_scaler.pkl'), 'wb') as f:
    pickle.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, f)
with open(os.path.join(MODEL_DIR, 'lstm_feature_scaler.pkl'), 'wb') as f:
    pickle.dump(feature_scaler, f)
with open(os.path.join(MODEL_DIR, 'lstm_target_scaler.pkl'), 'wb') as f:
    pickle.dump(target_scaler, f)

log = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'epochs': len(history.history['loss']),
       'lookback': LOOKBACK, 'features': feature_cols}
with open(os.path.join(LOGS_DIR, 'lstm_training_log.json'), 'w') as f:
    json.dump(log, f, indent=2, default=str)

print("\nLSTM model and scalers saved!")

if __name__ == '__main__':
    print("\nLSTM training complete!")
