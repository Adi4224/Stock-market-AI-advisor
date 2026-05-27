# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %%
# Imports and path configuration
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, add_targets

# Define output directories
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "reports", "training_plots")
LOGS_DIR = os.path.join(PROJECT_ROOT, "reports", "training_logs")
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# %%
# Load the raw stock dataset
data_path = os.path.join(PROJECT_ROOT, "data", "processed", "final_stock_dataset_2026.csv")
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully. Shape: {df.shape}")

# %%
# Process features ticker by ticker to avoid cross-contamination
df = add_technical_features(df)
df = add_targets(df)
print(f"Features engineered. Shape after additions: {df.shape}")

# %%
# Clean infinite and NaN values on the feature and target subset
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
    'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi',
    'macd', 'macd_signal', 'volume_change', 'price_range'
]
target_col = 'next_day_close'
subset_cols = feature_cols + [target_col]

# Replace infinite values with NaN and drop rows with NaN in subset
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=subset_cols).reset_index(drop=True)
print(f"Dataset cleaned. Shape after removing missing values: {df.shape}")

# %%
# Split dataset chronologically to prevent data leakage
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Determine the 80/20 split index
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# %%
# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# %%
# Train XGBoost Regressor using GridSearchCV
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1]
}

tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print(f"Grid search complete. Best parameters: {grid_search.best_params_}")

# %%
# Evaluate the model on test split
y_pred = best_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# %%
# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test[:200], label='Actual Close Price', color='#00D4FF', alpha=0.8, linewidth=2)
plt.plot(y_pred[:200], label='Predicted Close Price', color='#FF5252', alpha=0.8, linestyle='--', linewidth=2)
plt.title('XGBoost Regressor - Actual vs Predicted Close Prices (First 200 Test Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Stock Price ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plot_path = os.path.join(PLOTS_DIR, "xgb_actual_vs_predicted.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Actual vs Predicted plot saved to: {plot_path}")

# %%
# Save the trained XGBoost model and metrics log
model_path = os.path.join(MODEL_DIR, "xgboost_regressor.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

metrics = {
    'mae': round(float(mae), 4),
    'mse': round(float(mse), 4),
    'rmse': round(float(rmse), 4),
    'r2': round(float(r2), 4),
    'best_params': grid_search.best_params_
}

log_path = os.path.join(LOGS_DIR, "xgb_training_log.json")
with open(log_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Training log saved to: {log_path}")
