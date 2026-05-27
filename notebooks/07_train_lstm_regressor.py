# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %% [markdown]
# # 07 — Deep Learning LSTM Regressor Model Training
# Train a Long Short-Term Memory (LSTM) network to predict future stock prices.
# Uses Keras 3 with a PyTorch backend to enable high performance and native model saving.

# %%
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set the Keras backend to PyTorch before importing Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Keras Version: {keras.__version__}")
print(f"Backend: {keras.config.backend()}")

# %% [markdown]
# ## 1. Load Data
# Load raw stock dataset containing time-series observations for multiple tickers.

# %%
data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Processed dataset not found at: {data_path}")

df_raw = pd.read_csv(data_path)
print(f"Loaded raw dataset with shape: {df_raw.shape}")
print(f"Tickers present: {df_raw['Ticker'].nunique()}")

# %% [markdown]
# ## 2. Standalone Ticker-by-Ticker Feature Engineering
# Calculate the exact 16 technical features for stock prediction ticker by ticker to prevent data contamination.

# %%
def add_technical_features(df):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    result_frames = []
    for ticker, group in df.groupby('Ticker', sort=False):
        group = group.copy()
        
        # Returns
        group['daily_return'] = group['Close'].pct_change()
        
        # Moving Averages
        group['ma_7'] = group['Close'].rolling(window=7, min_periods=1).mean()
        group['ma_14'] = group['Close'].rolling(window=14, min_periods=1).mean()
        group['ma_30'] = group['Close'].rolling(window=30, min_periods=1).mean()
        group['ma_50'] = group['Close'].rolling(window=50, min_periods=1).mean()
        
        # Volatility (20-day rolling std of daily return)
        group['volatility'] = group['daily_return'].rolling(window=20, min_periods=1).std()
        
        # RSI (14-day manual)
        delta = group['Close'].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        group['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume change
        group['volume_change'] = group['Volume'].pct_change()
        
        # Price range
        group['price_range'] = (group['High'] - group['Low']) / group['Close']
        
        result_frames.append(group)
        
    return pd.concat(result_frames, ignore_index=True)

df_features = add_technical_features(df_raw)

# Target Variable is the Close price
target_col = 'Close'
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
    'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi',
    'macd', 'macd_signal', 'volume_change', 'price_range'
]

# %% [markdown]
# ## 3. Data Cleaning
# Clean infinite and NaN values properly specifically on the feature + target subset.

# %%
df_features = df_features.replace([np.inf, -np.inf], np.nan)
df_clean = df_features.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
print(f"Cleaned dataset shape: {df_clean.shape}")

# %% [markdown]
# ## 4. Chronological Train-Test Split (Ticker-by-Ticker)
# Perform an 80/20 chronological split ticker by ticker to prevent lookahead bias and data leakage.

# %%
train_frames = []
test_frames = []

for ticker, group in df_clean.groupby('Ticker', sort=False):
    group = group.sort_values('Date').reset_index(drop=True) if 'Date' in group.columns else group
    split_idx = int(len(group) * 0.8)
    
    train_frames.append(group.iloc[:split_idx])
    test_frames.append(group.iloc[split_idx:])

train_df = pd.concat(train_frames, ignore_index=True)
test_df = pd.concat(test_frames, ignore_index=True)

print(f"Train samples: {train_df.shape[0]} | Test samples: {test_df.shape[0]}")

# %% [markdown]
# ## 5. Scaling Features & Targets
# Scale the data using StandardScaler, fitting only on the training subset.

# %%
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Fit scaler on training subset
X_train_raw = train_df[feature_cols].values
y_train_raw = train_df[[target_col]].values

X_test_raw = test_df[feature_cols].values
y_test_raw = test_df[[target_col]].values

# Fit and transform
X_train_scaled = feature_scaler.fit_transform(X_train_raw)
y_train_scaled = target_scaler.fit_transform(y_train_raw).flatten()

# Transform testing set
X_test_scaled = feature_scaler.transform(X_test_raw)
y_test_scaled = target_scaler.transform(y_test_raw).flatten()

# Save scaled values back into temporary dataframes for sequence creation
train_scaled_df = train_df.copy()
train_scaled_df[feature_cols] = X_train_scaled
train_scaled_df[target_col] = y_train_scaled

test_scaled_df = test_df.copy()
test_scaled_df[feature_cols] = X_test_scaled
test_scaled_df[target_col] = y_test_scaled

# %% [markdown]
# ## 6. 3D Sequence Creation for LSTM (Lookback = 60)
# Create time-series sequences of lookback=60 ticker by ticker to avoid sequence overlap cross-contamination.

# %%
def create_sequences(data, target, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

LOOKBACK = 60
X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []

for ticker, group in train_scaled_df.groupby('Ticker', sort=False):
    if len(group) > LOOKBACK:
        X_seq, y_seq = create_sequences(group[feature_cols].values, group[target_col].values, LOOKBACK)
        X_train_list.append(X_seq)
        y_train_list.append(y_seq)

for ticker, group in test_scaled_df.groupby('Ticker', sort=False):
    if len(group) > LOOKBACK:
        X_seq, y_seq = create_sequences(group[feature_cols].values, group[target_col].values, LOOKBACK)
        X_test_list.append(X_seq)
        y_test_list.append(y_seq)

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Subsample training data to ensure rapid training on CPU while keeping representation of all 50 stocks
X_train_sub = X_train[::5]
y_train_sub = y_train[::5]

print(f"Full Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Subsampled Train shape: X={X_train_sub.shape}, y={y_train_sub.shape}")
print(f"Test shape: X={X_test.shape}, y={y_test.shape}")

# %% [markdown]
# ## 7. Build and Train stacked LSTM Regressor Model
# Initialize and compile the deep learning network. We use a stacked LSTM with 32 units to ensure rapid convergence.

# %%
model = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    LSTM(units=32, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Use Early Stopping and learning rate scheduler
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
]

# Train the LSTM model
history = model.fit(
    X_train_sub, y_train_sub,
    validation_split=0.1,
    epochs=5,
    batch_size=1024,
    callbacks=callbacks,
    verbose=1
)

# %% [markdown]
# ## 8. Model Evaluation
# Predict on the test split, apply inverse scaling to predictions and actuals, and compute validation metrics.

# %%
# Predict on test split
y_pred_scaled = model.predict(X_test, batch_size=2048, verbose=1).flatten()

# Inverse transform predictions and actual values back to original price scale
y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_predicted = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate regression evaluation metrics
mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_predicted)

print("\n" + "="*40)
print("LSTM Regressor Evaluation Metrics:")
print(f"  MAE:  {mae:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print("="*40)

# %% [markdown]
# ## 9. Save Model and Scalers
# Persist the trained model, scalers dictionary, and training log to the filesystem.

# %%
model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))

# Save standard scalers dictionary
scaler_dict = {
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler
}
with open(os.path.join(MODEL_DIR, 'lstm_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_dict, f)

# Also save individual scalers for maximum system compatibility
with open(os.path.join(MODEL_DIR, 'lstm_feature_scaler.pkl'), 'wb') as f:
    pickle.dump(feature_scaler, f)
with open(os.path.join(MODEL_DIR, 'lstm_target_scaler.pkl'), 'wb') as f:
    pickle.dump(target_scaler, f)

# Save training log as JSON
log_data = {
    'model_name': 'LSTM Regressor',
    'mae': float(mae),
    'mse': float(mse),
    'rmse': float(rmse),
    'r2': float(r2),
    'epochs_trained': len(history.history['loss']),
    'lookback': LOOKBACK,
    'features': feature_cols
}

with open(os.path.join(LOGS_DIR, 'lstm_training_log.json'), 'w') as f:
    json.dump(log_data, f, indent=2)

print("\nModel training, scaling, and logging completed successfully!")

if __name__ == '__main__':
    print("\nLSTM Model Training Execution Finished!")
