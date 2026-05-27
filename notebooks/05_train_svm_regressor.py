# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %% [markdown]
# # SVM Regressor Model Training
# Train Support Vector Regression (SVR) for stock price prediction.

# %%
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup project directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, add_targets

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
# Load the processed dataset
data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"Dataset not found at {data_path}")

print(f"Raw data shape: {df.shape}")

# %% [markdown]
# ## 2. Feature Engineering & Preprocessing

# %%
# Add technical features and target variable ticker by ticker to avoid cross-contamination
print("Adding technical features and targets...")
df = add_technical_features(df)
df = add_targets(df)

# Filter for a single ticker to allow efficient SVR training
if 'Ticker' in df.columns:
    ticker = df['Ticker'].value_counts().index[0]
    df = df[df['Ticker'] == ticker].copy()
    print(f"Filtered for primary ticker: {ticker}")

# Define exactly 16 technical features
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'daily_return', 'ma_7', 'ma_14', 'ma_30', 'ma_50',
    'volatility', 'rsi', 'macd', 'macd_signal', 'volume_change', 'price_range'
]

# Clean infinite/NaN values specifically on feature + target subset
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ['next_day_close']).reset_index(drop=True)
print(f"Processed dataset shape: {df.shape}")

# %% [markdown]
# ## 3. Train-Test Split & Scaling

# %%
# Split data into train and test sets (80/20 chronological split)
split_idx = int(len(df) * 0.8)
X_train = df[feature_cols].iloc[:split_idx].values
X_test = df[feature_cols].iloc[split_idx:].values
y_train = df['next_day_close'].iloc[:split_idx].values
y_test = df['next_day_close'].iloc[split_idx:].values

# Scale features using StandardScaler
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale target using StandardScaler (essential for SVR stability)
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")

# %% [markdown]
# ## 4. Train SVM Regressor

# %%
# Hyperparameter grid for SVR tuning
param_grid = {
    'kernel': ['rbf'],
    'C': [1, 10, 100],
    'gamma': ['scale'],
    'epsilon': [0.05, 0.1]
}

# Grid search with TimeSeriesSplit (using n_jobs=1 to guarantee no multiprocessing deadlock)
print("Starting GridSearchCV training (n_jobs=1)...")
tscv = TimeSeriesSplit(n_splits=3)
svr_grid = GridSearchCV(
    SVR(),
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=2
)

svr_grid.fit(X_train_scaled, y_train_scaled)
best_model = svr_grid.best_estimator_
print(f"Grid search complete. Best hyperparameters: {svr_grid.best_params_}")

# %% [markdown]
# ## 5. Model Evaluation

# %%
# Generate predictions on scaled test set
print("Generating predictions and calculating metrics...")
y_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform predictions and actual values back to original price scale
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Compute regression evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nSVM Regressor Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# %% [markdown]
# ## 6. Visualizations

# %%
# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test[:150], label='Actual Close', color='#3B82F6', linewidth=2)
plt.plot(y_pred[:150], label='Predicted Close', color='#EF4444', linewidth=2, linestyle='--')
plt.title('SVM Regressor - Actual vs Predicted Stock Prices', fontsize=14)
plt.xlabel('Time Step')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'svm_actual_vs_predicted.png'), dpi=150)
print("Saved actual vs predicted stock price plot.")

# %% [markdown]
# ## 7. Save Model and Scalers

# %%
# Save the SVR model and scalers to the models directory
joblib.dump(best_model, os.path.join(MODEL_DIR, 'svm_regressor.pkl'))
joblib.dump(feature_scaler, os.path.join(MODEL_DIR, 'svm_scaler.pkl'))
joblib.dump(target_scaler, os.path.join(MODEL_DIR, 'svm_target_scaler.pkl'))

# Save training logs in JSON format
log_data = {
    'model_name': 'SVM Regressor',
    'best_params': svr_grid.best_params_,
    'metrics': {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
}
with open(os.path.join(LOGS_DIR, 'svm_regressor_training_log.json'), 'w') as f:
    json.dump(log_data, f, indent=4)

print("Model, scalers, and training log saved successfully.")
