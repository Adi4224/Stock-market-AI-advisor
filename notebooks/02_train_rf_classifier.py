# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %% [markdown]
# # Stock Market AI Advisor - Random Forest Classifier Training
# This notebook trains a Random Forest Classifier to predict up/down stock movements.
# Guarded with standard if __name__ == '__main__' block to prevent Windows multiprocessing deadlocks.

# %%
# Import required libraries
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Define directory paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')

for directory in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# %%
# Process technical features and target variables ticker by ticker
def add_technical_indicators(df):
    processed_dfs = []
    # Group by Ticker and calculate rolling indicators to prevent cross-contamination
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date').copy()
        
        # Calculate returns
        group['daily_return'] = group['Close'].pct_change()
        
        # Moving averages
        group['ma_7'] = group['Close'].rolling(window=7, min_periods=1).mean()
        group['ma_14'] = group['Close'].rolling(window=14, min_periods=1).mean()
        group['ma_30'] = group['Close'].rolling(window=30, min_periods=1).mean()
        group['ma_50'] = group['Close'].rolling(window=50, min_periods=1).mean()
        
        # Volatility (20-day rolling standard deviation of daily return)
        group['volatility'] = group['daily_return'].rolling(window=20, min_periods=1).std()
        
        # Relative Strength Index (RSI, 14-day)
        delta = group['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        group['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12-day EMA minus 26-day EMA of Close) and MACD Signal (9-day EMA of MACD)
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume change
        group['volume_change'] = group['Volume'].pct_change()
        
        # Price range
        group['price_range'] = (group['High'] - group['Low']) / group['Close']
        
        # Classification Target (1 if next close is higher than current close, 0 otherwise)
        group['next_day_close'] = group['Close'].shift(-1)
        group['movement'] = (group['next_day_close'] > group['Close']).astype(int)
        
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

# %%
def run_training_pipeline():
    # Load stock price history dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found at: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded raw dataset with shape: {df_raw.shape}")

    df_features = add_technical_indicators(df_raw)
    print(f"Dataset shape after technical indicators: {df_features.shape}")

    # Clean infinite and NaN values properly from target and features
    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
        'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi',
        'macd', 'macd_signal', 'volume_change', 'price_range'
    ]
    TARGET_COL = 'movement'

    # Replace any inf values with NaN
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop NaN specifically on the subset of feature + target columns
    df_cleaned = df_features.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"Dataset shape after dropping NaN values: {df_cleaned.shape}")

    # Perform chronological time-based split to avoid data leakage
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_sorted = df_cleaned.sort_values(by='Date').reset_index(drop=True)

    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    print(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

    # Scale technical features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaled feature transformer
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Fitted StandardScaler saved to: {scaler_path}")

    # Train Random Forest Classifier using GridSearchCV and TimeSeriesSplit
    # Downsample training split specifically for Grid Search to ensure fast execution
    MAX_GRID_SAMPLES = 10000
    if len(X_train_scaled) > MAX_GRID_SAMPLES:
        X_grid = X_train_scaled[-MAX_GRID_SAMPLES:]
        y_grid = y_train[-MAX_GRID_SAMPLES:]
    else:
        X_grid = X_train_scaled
        y_grid = y_train

    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    rf_base = RandomForestClassifier(random_state=42, n_jobs=1)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=1,  # Set n_jobs=1 to avoid deadlocks on Windows
        verbose=1
    )

    print("Starting Random Forest Classifier training via GridSearchCV...")
    grid_search.fit(X_grid, y_grid)

    best_model = grid_search.best_estimator_
    print(f"Grid search complete. Best parameters found: {grid_search.best_params_}")

    # Fit the best model on full training dataset (takes < 15 seconds)
    print("Fitting final Random Forest Classifier model on full training split...")
    best_model.fit(X_train_scaled, y_train)
    print("Final model fit complete.")

    # Predict and evaluate classification metrics on the test split
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n===== Random Forest Classifier Performance =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("================================================")

    # Generate confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('Random Forest Classifier - Confusion Matrix')
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, 'rf_classifier_confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_plot_path}")

    # Generate feature importance plot
    importances = best_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in indices])
    ax.set_xlabel('Relative Feature Importance')
    ax.set_title('Random Forest Classifier - Feature Importances')
    plt.tight_layout()
    fi_plot_path = os.path.join(PLOTS_DIR, 'rf_classifier_feature_importance.png')
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to: {fi_plot_path}")

    # Save the final best estimator and training metrics logs
    model_save_path = os.path.join(MODEL_DIR, 'random_forest_classifier.pkl')
    joblib.dump(best_model, model_save_path)
    print(f"Best Random Forest Classifier model saved to: {model_save_path}")

    training_metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'best_params': grid_search.best_params_
    }

    log_save_path = os.path.join(LOGS_DIR, 'rf_classifier_training_log.json')
    with open(log_save_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    print(f"Training metrics log saved to: {log_save_path}")

# %%
if __name__ == '__main__':
    run_training_pipeline()
    print("Random Forest Classifier model training process finished successfully.")
