# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %% [markdown]
# # 06 — Support Vector Machine Classifier Model Training
# Train a Support Vector Classifier (SVC) for stock price movement prediction.

# %%
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Set up project root and directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## 1. Feature Engineering and Preparation Functions

# %%
def add_technical_features(df):
    """Add technical indicator columns to a stock DataFrame."""
    df = df.copy()

    # Helper to compute RSI for a single series
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # Process each ticker independently to avoid cross-contamination
    result_frames = []
    for ticker, group in df.groupby('Ticker', sort=False):
        group = group.sort_values('Date').copy() if 'Date' in group.columns else group.copy()

        # Daily return
        group['daily_return'] = group['Close'].pct_change()

        # Moving averages
        group['ma_7'] = group['Close'].rolling(window=7, min_periods=1).mean()
        group['ma_14'] = group['Close'].rolling(window=14, min_periods=1).mean()
        group['ma_30'] = group['Close'].rolling(window=30, min_periods=1).mean()
        group['ma_50'] = group['Close'].rolling(window=50, min_periods=1).mean()

        # Volatility (20-day rolling std of daily return)
        group['volatility'] = group['daily_return'].rolling(window=20, min_periods=1).std()

        # RSI (14-day)
        group['rsi'] = _compute_rsi(group['Close'], period=14)

        # MACD
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()

        # Volume change
        group['volume_change'] = group['Volume'].pct_change()

        # Price range
        group['price_range'] = (group['High'] - group['Low']) / group['Close']

        result_frames.append(group)

    df = pd.concat(result_frames, ignore_index=True)
    return df

def add_targets(df):
    """Add target columns for supervised learning."""
    df = df.copy()
    df['next_day_close'] = df.groupby('Ticker')['Close'].shift(-1)
    df['movement'] = (df['next_day_close'] > df['Close']).astype(int)
    return df

# %% [markdown]
# ## 2. Main Training Execution Function

# %%
def run_training_pipeline():
    """Execute the full dataset loading, scaling, training, and evaluation pipeline."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')

    if os.path.exists(data_path):
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found locally. Generating synthetic fallback data for training.")
        dates = pd.date_range(start="2020-01-01", end="2026-01-01", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        frames = []
        np.random.seed(42)
        for ticker in tickers:
            price = 100.0
            prices = []
            volumes = []
            for _ in range(len(dates)):
                price *= (1.0 + np.random.normal(0.0005, 0.015))
                prices.append(price)
                volumes.append(int(np.random.normal(1000000, 200000)))
            
            t_df = pd.DataFrame({
                "Date": dates,
                "Open": [p * (1.0 - 0.005) for p in prices],
                "High": [p * (1.0 + 0.01) for p in prices],
                "Low": [p * (1.0 - 0.01) for p in prices],
                "Close": prices,
                "Volume": volumes,
                "Ticker": ticker
            })
            frames.append(t_df)
        df = pd.concat(frames, ignore_index=True)

    print(f"Raw data shape: {df.shape}")

    # %% [markdown]
    # ## 3. Compute Features and Targets
    print("Computing features and targets...")
    df_feat = add_technical_features(df)
    df_feat = add_targets(df_feat)

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
        'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi',
        'macd', 'macd_signal', 'volume_change', 'price_range'
    ]

    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(subset=feature_cols + ['movement'], inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    print(f"Cleaned dataset shape: {df_feat.shape}")
    print(f"Available feature columns: {len(feature_cols)}")

    # %% [markdown]
    # ## 4. Chronological Train-Test Split
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    df_feat = df_feat.sort_values('Date').reset_index(drop=True)

    split_idx = int(len(df_feat) * 0.8)
    X_train = df_feat[feature_cols].iloc[:split_idx].values
    X_test = df_feat[feature_cols].iloc[split_idx:].values
    y_train = df_feat['movement'].iloc[:split_idx].values
    y_test = df_feat['movement'].iloc[split_idx:].values

    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
    print(f"Train labels distribution: {np.bincount(y_train)}")
    print(f"Test labels distribution: {np.bincount(y_test)}")

    # %% [markdown]
    # ## 5. Standard Scaling of Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to: {scaler_path}")

    # %% [markdown]
    # ## 6. Hyperparameter Optimization via TimeSeriesSplit
    # Downsample training set specifically for SVC grid search to avoid deadlock/hang
    MAX_GRID_SAMPLES = 2000
    if len(X_train_scaled) > MAX_GRID_SAMPLES:
        X_grid = X_train_scaled[-MAX_GRID_SAMPLES:]
        y_grid = y_train[-MAX_GRID_SAMPLES:]
    else:
        X_grid = X_train_scaled
        y_grid = y_train

    param_grid = {
        'C': [1.0, 10.0],
        'gamma': ['scale']
    }

    print("Starting Grid Search CV on training subset...")
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        param_grid,
        cv=tscv,
        scoring='f1',
        n_jobs=1,  # Set n_jobs=1 to guarantee no deadlock on Windows
        verbose=1
    )
    grid_search.fit(X_grid, y_grid)
    best_params = grid_search.best_params_
    print(f"Best hyperparameters found: {best_params}")

    # %% [markdown]
    # ## 7. Final Model Training on Downsampled Dataset
    MAX_TRAIN_SAMPLES = 10000
    if len(X_train_scaled) > MAX_TRAIN_SAMPLES:
        np.random.seed(42)
        indices = np.random.choice(len(X_train_scaled), size=MAX_TRAIN_SAMPLES, replace=False)
        X_train_fit = X_train_scaled[indices]
        y_train_fit = y_train[indices]
    else:
        X_train_fit = X_train_scaled
        y_train_fit = y_train

    print(f"Training final Support Vector Classifier on: {X_train_fit.shape[0]} samples")
    model = SVC(
        kernel='rbf',
        C=best_params['C'],
        gamma=best_params['gamma'],
        probability=True,
        random_state=42
    )
    model.fit(X_train_fit, y_train_fit)
    print("Model training complete.")

    # %% [markdown]
    # ## 8. Model Evaluation on Test Split
    # Downsample test split for evaluation to ensure prompt and efficient predictions
    MAX_TEST_SAMPLES = 5000
    if len(X_test_scaled) > MAX_TEST_SAMPLES:
        np.random.seed(42)
        test_indices = np.random.choice(len(X_test_scaled), size=MAX_TEST_SAMPLES, replace=False)
        X_test_eval = X_test_scaled[test_indices]
        y_test_eval = y_test[test_indices]
    else:
        X_test_eval = X_test_scaled
        y_test_eval = y_test

    y_pred = model.predict(X_test_eval)

    accuracy = accuracy_score(y_test_eval, y_pred)
    precision = precision_score(y_test_eval, y_pred, zero_division=0)
    recall = recall_score(y_test_eval, y_pred, zero_division=0)
    f1 = f1_score(y_test_eval, y_pred, zero_division=0)

    print("\nSupport Vector Classifier Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # %% [markdown]
    # ## 9. Save Confusion Matrix Plot and Model Artifacts
    model_path = os.path.join(MODEL_DIR, 'svm_classifier.pkl')
    joblib.dump(model, model_path)
    print(f"Saved SVM Classifier model to: {model_path}")

    cm = confusion_matrix(y_test_eval, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('SVC: Movement Prediction Confusion Matrix', fontsize=14)
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, 'svm_confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    log_data = {
        "model_name": "Support Vector Classifier",
        "best_parameters": best_params,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "features_used": feature_cols,
        "training_samples": X_train_fit.shape[0],
        "test_samples": X_test_eval.shape[0]
    }

    log_path = os.path.join(LOGS_DIR, 'svm_classifier_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Saved training log to: {log_path}")

# %%
if __name__ == '__main__':
    run_training_pipeline()
    print("\nSVM Classifier model training process finished successfully.")
