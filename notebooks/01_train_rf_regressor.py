# Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro

# %% [markdown]
# # Stock Market AI Advisor - Model Training
# Developed by: Adithya Dadi
# Program: Summer Internship - Agentic AI | DataPro
#
# This notebook implements the end-to-end training pipeline for the Random Forest Regressor.

# %%
# Import required libraries.
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
# Configure file paths and establish save directories.
data_path = 'data/processed/final_stock_dataset_2026.csv'
model_path = 'models/random_forest_regressor.pkl'
scaler_path = 'models/scaler.pkl'
notebook_py_path = 'notebooks/01_train_rf_regressor.py'
notebook_ipynb_path = 'notebooks/01_train_rf_regressor.ipynb'

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# %%
# Load the raw stock dataset.
print("Loading raw dataset...")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# %%
# Compute the 16 technical indicators ticker by ticker to avoid cross-contamination.
def compute_technical_indicators(df):
    processed_dfs = []
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    for ticker, group in df.groupby('Ticker', sort=False):
        group = group.sort_values('Date').copy()
        
        # Compute daily return
        group['daily_return'] = group['Close'].pct_change()
        
        # Compute moving averages
        group['ma_7'] = group['Close'].rolling(window=7, min_periods=1).mean()
        group['ma_14'] = group['Close'].rolling(window=14, min_periods=1).mean()
        group['ma_30'] = group['Close'].rolling(window=30, min_periods=1).mean()
        group['ma_50'] = group['Close'].rolling(window=50, min_periods=1).mean()
        
        # Compute volatility (20-day rolling standard deviation of daily_return)
        group['volatility'] = group['daily_return'].rolling(window=20, min_periods=1).std()
        
        # Compute Relative Strength Index (RSI-14)
        delta = group['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        group['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        
        # Compute MACD and MACD Signal
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
        
        # Compute volume change
        group['volume_change'] = group['Volume'].pct_change()
        
        # Compute price range
        group['price_range'] = (group['High'] - group['Low']) / group['Close']
        
        # Compute target column (next-day Close price)
        group['next_day_close'] = group['Close'].shift(-1)
        
        processed_dfs.append(group)
        
    return pd.concat(processed_dfs, ignore_index=True)

# %%
# Execute the ticker-by-ticker feature engineering function.
df_processed = compute_technical_indicators(df)
print(f"Processed dataset shape: {df_processed.shape}")

# %%
# Define feature columns and target variables.
features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 
    'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi', 
    'macd', 'macd_signal', 'volume_change', 'price_range'
]
target = 'next_day_close'

# %%
# Clean infinite and missing values specifically on features and target columns.
df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_count = len(df_processed)
df_clean = df_processed.dropna(subset=features + [target]).copy()
print(f"Cleaned dataset shape: {df_clean.shape} (dropped {initial_count - len(df_clean)} rows containing NaNs or Infs)")

# %%
# Sort the entire dataset chronologically to prepare for time-series splitting.
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean.sort_values('Date', inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# %%
# Chronologically split the dataset into 80% training and 20% testing data.
split_idx = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_idx]
test_df = df_clean.iloc[split_idx:]

X_train = train_df[features].values
y_train = train_df[target].values
X_test = test_df[features].values
y_test = test_df[target].values

print(f"Train shape - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test shape - X: {X_test.shape}, y: {y_test.shape}")

# %%
# Scale features using StandardScaler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Successfully scaled features.")

# %%
# Initialize and train the Random Forest Regressor model.
print("Training Random Forest Regressor model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("Model training completed.")

# %%
# Save the trained Random Forest model and standard scaler.
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")

# %%
# Evaluate the model on test split and compute regression metrics.
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Model Evaluation Metrics =====")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print("====================================")

# %%
# Export the Python script as a standard Jupyter Notebook file.
def export_to_notebook(py_src, ipynb_dest):
    with open(py_src, 'r', encoding='utf-8') as f:
        code_lines = f.read().split('\n')
        
    cells = []
    current_cell_content = []
    current_cell_type = 'code'
    
    for line in code_lines:
        if line.strip().startswith('# %% [markdown]'):
            if current_cell_content:
                cells.append((current_cell_type, '\n'.join(current_cell_content)))
                current_cell_content = []
            current_cell_type = 'markdown'
        elif line.strip().startswith('# %%'):
            if current_cell_content:
                cells.append((current_cell_type, '\n'.join(current_cell_content)))
                current_cell_content = []
            current_cell_type = 'code'
        else:
            if current_cell_type == 'markdown':
                if line.startswith('# '):
                    current_cell_content.append(line[2:])
                elif line.startswith('#'):
                    current_cell_content.append(line[1:].lstrip() if len(line) > 1 else '')
                else:
                    current_cell_content.append(line)
            else:
                current_cell_content.append(line)
                
    if current_cell_content:
        cells.append((current_cell_type, '\n'.join(current_cell_content)))
        
    notebook_dict = {
        'nbformat': 4,
        'nbformat_minor': 0,
        'metadata': {
            'colab': {'provenance': [], 'name': os.path.basename(ipynb_dest)},
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'}
        },
        'cells': []
    }
    
    for cell_type, content_str in cells:
        content_str = content_str.strip()
        if not content_str:
            continue
        cell_obj = {
            'cell_type': cell_type,
            'metadata': {},
            'source': [l + '\n' for l in content_str.split('\n')]
        }
        if cell_obj['source']:
            cell_obj['source'][-1] = cell_obj['source'][-1].rstrip('\n')
        if cell_type == 'code':
            cell_obj['execution_count'] = None
            cell_obj['outputs'] = []
        notebook_dict['cells'].append(cell_obj)
        
    with open(ipynb_dest, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
        
    print(f"Jupyter Notebook successfully created at: {ipynb_dest}")

export_to_notebook(notebook_py_path, notebook_ipynb_path)
