# %% [markdown]
# # Stock Market AI Advisor - Model Training
# # Developed by: Adithya Dadi
# # Program: Summer Internship - Agentic AI | DataPro
#
# # %%
# # Import libraries
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
# Define project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, add_targets
from src.backend.stock_preprocessing import preprocess_stock_data

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# %%
# Load dataset
data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Raw data shape: {df.shape}")

# %%
# Feature engineering and target creation ticker by ticker
print("Adding technical features and targets ticker by ticker...")
df = preprocess_stock_data(df)
df = add_technical_features(df)
df = add_targets(df)

# %%
# Clean infinite and NaN values on feature and target subset
print("Cleaning infinite and NaN values...")
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 
    'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi', 
    'macd', 'macd_signal', 'volume_change', 'price_range'
]
target_col = 'movement'

# Replace +/- inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop NaN values in features and target
df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
print(f"Cleaned dataset shape: {df.shape}")

# %%
# Split data into train and test sets chronologically
print("Splitting dataset chronologically...")
# Sort by Date to ensure chronological split across tickers
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)

# 80/20 Train/Test split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# %%
# Scale features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# %%
# Hyperparameter tuning with GridSearchCV and TimeSeriesSplit
print("Training XGBoost Classifier with hyperparameter tuning...")
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8]
}

xgb = XGBClassifier(
    random_state=42, 
    verbosity=0, 
    use_label_encoder=False, 
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    cv=tscv, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# %%
# Evaluate the model on test set
print("Evaluating the model on test split...")
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# %%
# Plot and save confusion matrix
print("Generating evaluation plots...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down/Flat', 'Up'], yticklabels=['Down/Flat', 'Up'])
plt.title('XGBoost Classifier - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, 'xgb_classifier_confusion_matrix.png')
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Confusion matrix plot saved to {plot_path}")

# %%
# Save the model and training logs
model_path = os.path.join(MODEL_DIR, 'xgboost_classifier.pkl')
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

log_path = os.path.join(LOGS_DIR, 'xgb_classifier_training_log.json')
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'best_params': grid_search.best_params_
}
with open(log_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Training log saved to {log_path}")

if __name__ == '__main__':
    print("XGBoost Classifier training pipeline complete.")
