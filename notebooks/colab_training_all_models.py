# %% [markdown]
# # Stock Market AI Advisor - Model Training
# Adithya Dadi | Summer Internship - Agentic AI, DataPro

# %% [markdown]
# ## Install Dependencies

# %%
!pip install -q pandas numpy scikit-learn xgboost yfinance matplotlib seaborn joblib

# %% [markdown]
# ## Fetch Stock Data

# %%
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FETCHING STOCK DATA")
print("=" * 60)

tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
frames = []

for t in tickers:
    ticker_obj = yf.Ticker(t)
    df = ticker_obj.history(period='5y')
    if df is not None and not df.empty:
        df = df.reset_index()
        df.columns = [c.strip().title() for c in df.columns]
        df['Ticker'] = t
        frames.append(df)
        print(f"  {t}: {len(df)} rows ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")

data = pd.concat(frames, ignore_index=True)
print(f"\nTotal dataset: {len(data)} rows, {data['Ticker'].nunique()} tickers")
data.head()

# %% [markdown]
# ## Feature Engineering

# %%
def add_features(df):
    """Add technical indicators and target variables."""
    df = df.sort_values('Date').copy()

    # Price features
    df['daily_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Moving averages
    for w in [7, 14, 30, 50, 100]:
        df[f'ma_{w}'] = df['Close'].rolling(w).mean()

    # Volatility
    df['volatility'] = df['daily_return'].rolling(20).std()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Volume features
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma_20'] = df['Volume'].rolling(20).mean()

    # Price range
    df['price_range'] = (df['High'] - df['Low']) / df['Close']

    # Targets
    df['next_day_close'] = df['Close'].shift(-1)
    df['movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df

# Process each ticker
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return',
                'ma_7', 'ma_14', 'ma_30', 'ma_50', 'volatility', 'rsi',
                'macd', 'macd_signal', 'volume_change', 'price_range']

# Use AAPL for training (single ticker for cleaner signals)
train_ticker = 'AAPL'
df = data[data['Ticker'] == train_ticker].copy()
df = add_features(df)
df = df.dropna().reset_index(drop=True)

available_features = [c for c in FEATURE_COLS if c in df.columns]

print(f"\nFeature engineering complete for {train_ticker}")
print(f"   Rows: {len(df)}")
print(f"   Features ({len(available_features)}): {available_features}")

# %% [markdown]
# ## Train-Test Split and Scaling

# %%
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Time-based 80/20 split
split = int(len(df) * 0.8)

X_train = df[available_features].iloc[:split].values
X_test = df[available_features].iloc[split:].values
y_train_reg = df['next_day_close'].iloc[:split].values
y_test_reg = df['next_day_close'].iloc[split:].values
y_train_cls = df['movement'].iloc[:split].values
y_test_cls = df['movement'].iloc[split:].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train-Test Split:")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")
print(f"   Features: {X_train.shape[1]}")
print(f"   Train date range: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[split].strftime('%Y-%m-%d')}")
print(f"   Test date range:  {df['Date'].iloc[split].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

# %% [markdown]
# ## Model 1: Random Forest

# %%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

print("=" * 60)
print("TRAINING: Random Forest Regressor")
print("=" * 60)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
}
tscv = TimeSeriesSplit(n_splits=3)

rf_reg = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid,
                      cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_reg.fit(X_train_scaled, y_train_reg)

y_pred_rf_reg = rf_reg.predict(X_test_scaled)

rf_reg_mae = mean_absolute_error(y_test_reg, y_pred_rf_reg)
rf_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf_reg))
rf_reg_r2 = r2_score(y_test_reg, y_pred_rf_reg)

print(f"\nRandom Forest Regressor Results:")
print(f"   Best params: {rf_reg.best_params_}")
print(f"   MAE:  {rf_reg_mae:.4f}")
print(f"   RMSE: {rf_reg_rmse:.4f}")
print(f"   R2:   {rf_reg_r2:.4f}")

# %%
print("=" * 60)
print("TRAINING: Random Forest Classifier")
print("=" * 60)

rf_cls_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
}
rf_cls = GridSearchCV(RandomForestClassifier(random_state=42), rf_cls_params,
                      cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
rf_cls.fit(X_train_scaled, y_train_cls)

y_pred_rf_cls = rf_cls.predict(X_test_scaled)

rf_cls_acc = accuracy_score(y_test_cls, y_pred_rf_cls)
rf_cls_f1 = f1_score(y_test_cls, y_pred_rf_cls)

print(f"\nRandom Forest Classifier Results:")
print(f"   Best params: {rf_cls.best_params_}")
print(f"   Accuracy:  {rf_cls_acc:.4f}")
print(f"   Precision: {precision_score(y_test_cls, y_pred_rf_cls):.4f}")
print(f"   Recall:    {recall_score(y_test_cls, y_pred_rf_cls):.4f}")
print(f"   F1 Score:  {rf_cls_f1:.4f}")

# %% [markdown]
# ## Model 2: XGBoost

# %%
from xgboost import XGBRegressor, XGBClassifier

print("=" * 60)
print("TRAINING: XGBoost Regressor")
print("=" * 60)

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
}
xgb_reg = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), xgb_param_grid,
                       cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_reg.fit(X_train_scaled, y_train_reg)

y_pred_xgb_reg = xgb_reg.predict(X_test_scaled)

xgb_reg_mae = mean_absolute_error(y_test_reg, y_pred_xgb_reg)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb_reg))
xgb_reg_r2 = r2_score(y_test_reg, y_pred_xgb_reg)

print(f"\nXGBoost Regressor Results:")
print(f"   Best params: {xgb_reg.best_params_}")
print(f"   MAE:  {xgb_reg_mae:.4f}")
print(f"   RMSE: {xgb_reg_rmse:.4f}")
print(f"   R2:   {xgb_reg_r2:.4f}")

# %%
print("=" * 60)
print("TRAINING: XGBoost Classifier")
print("=" * 60)

xgb_cls_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
}
xgb_cls = GridSearchCV(XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss'),
                       xgb_cls_params, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
xgb_cls.fit(X_train_scaled, y_train_cls)

y_pred_xgb_cls = xgb_cls.predict(X_test_scaled)

xgb_cls_acc = accuracy_score(y_test_cls, y_pred_xgb_cls)
xgb_cls_f1 = f1_score(y_test_cls, y_pred_xgb_cls)

print(f"\nXGBoost Classifier Results:")
print(f"   Best params: {xgb_cls.best_params_}")
print(f"   Accuracy:  {xgb_cls_acc:.4f}")
print(f"   Precision: {precision_score(y_test_cls, y_pred_xgb_cls):.4f}")
print(f"   Recall:    {recall_score(y_test_cls, y_pred_xgb_cls):.4f}")
print(f"   F1 Score:  {xgb_cls_f1:.4f}")

# %% [markdown]
# ## Model 3: Support Vector Machine

# %%
from sklearn.svm import SVR, SVC

print("=" * 60)
print("TRAINING: Support Vector Regressor (SVR)")
print("=" * 60)

# Scale target for SVR
target_scaler = StandardScaler()
y_train_reg_scaled = target_scaler.fit_transform(y_train_reg.reshape(-1, 1)).flatten()

svr_param_grid = {
    'kernel': ['rbf'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.5],
}
svr = GridSearchCV(SVR(), svr_param_grid, cv=tscv,
                   scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
svr.fit(X_train_scaled, y_train_reg_scaled)

y_pred_svr_scaled = svr.predict(X_test_scaled)
y_pred_svr = target_scaler.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).flatten()

svr_mae = mean_absolute_error(y_test_reg, y_pred_svr)
svr_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_svr))
svr_r2 = r2_score(y_test_reg, y_pred_svr)

print(f"\nSVR Results:")
print(f"   Best params: {svr.best_params_}")
print(f"   MAE:  {svr_mae:.4f}")
print(f"   RMSE: {svr_rmse:.4f}")
print(f"   R2:   {svr_r2:.4f}")

# %%
print("=" * 60)
print("TRAINING: Support Vector Classifier (SVC)")
print("=" * 60)

svc_param_grid = {
    'kernel': ['rbf'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto'],
}
svc = GridSearchCV(SVC(probability=True), svc_param_grid, cv=tscv,
                   scoring='f1', n_jobs=-1, verbose=1)
svc.fit(X_train_scaled, y_train_cls)

y_pred_svc = svc.predict(X_test_scaled)

svc_acc = accuracy_score(y_test_cls, y_pred_svc)
svc_f1 = f1_score(y_test_cls, y_pred_svc)

print(f"\nSVC Results:")
print(f"   Best params: {svc.best_params_}")
print(f"   Accuracy:  {svc_acc:.4f}")
print(f"   Precision: {precision_score(y_test_cls, y_pred_svc):.4f}")
print(f"   Recall:    {recall_score(y_test_cls, y_pred_svc):.4f}")
print(f"   F1 Score:  {svc_f1:.4f}")

# %% [markdown]
# ## Model Comparison

# %%
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

# Regression comparison
print("\nREGRESSION RESULTS (Stock Price Prediction)")
print("-" * 55)
print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
print("-" * 55)
print(f"{'Random Forest':<25} {rf_reg_mae:>10.4f} {rf_reg_rmse:>10.4f} {rf_reg_r2:>10.4f}")
print(f"{'XGBoost':<25} {xgb_reg_mae:>10.4f} {xgb_reg_rmse:>10.4f} {xgb_reg_r2:>10.4f}")
print(f"{'SVR':<25} {svr_mae:>10.4f} {svr_rmse:>10.4f} {svr_r2:>10.4f}")
print("-" * 55)

# Classification comparison
print("\nCLASSIFICATION RESULTS (Movement Prediction)")
print("-" * 55)
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10}")
print("-" * 55)
print(f"{'Random Forest':<25} {rf_cls_acc:>10.4f} {rf_cls_f1:>10.4f}")
print(f"{'XGBoost':<25} {xgb_cls_acc:>10.4f} {xgb_cls_f1:>10.4f}")
print(f"{'SVC':<25} {svc_acc:>10.4f} {svc_f1:>10.4f}")
print("-" * 55)

# Best model selection
reg_models = {'Random Forest': rf_reg_rmse, 'XGBoost': xgb_reg_rmse, 'SVR': svr_rmse}
cls_models = {'Random Forest': rf_cls_f1, 'XGBoost': xgb_cls_f1, 'SVC': svc_f1}

best_reg = min(reg_models, key=reg_models.get)
best_cls = max(cls_models, key=cls_models.get)

print(f"\nBest Regression Model:     {best_reg} (RMSE: {reg_models[best_reg]:.4f})")
print(f"Best Classification Model: {best_cls} (F1: {cls_models[best_cls]:.4f})")

# %% [markdown]
# ## Actual vs Predicted - All Models

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

models_pred = [
    ('Random Forest', y_pred_rf_reg, '#818CF8'),
    ('XGBoost', y_pred_xgb_reg, '#34D399'),
    ('SVR', y_pred_svr, '#FBBF24'),
]

for ax, (name, y_pred, color) in zip(axes, models_pred):
    ax.plot(y_test_reg[:150], label='Actual', color='#818CF8', linewidth=1.5, alpha=0.7)
    ax.plot(y_pred[:150], label='Predicted', color=color, linewidth=1.5, linestyle='--')
    ax.set_title(f'{name}: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_regression.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison_regression.png")

# %% [markdown]
# ## Confusion Matrices

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cls_preds = [
    ('Random Forest', y_pred_rf_cls),
    ('XGBoost', y_pred_xgb_cls),
    ('SVC', y_pred_svc),
]

for ax, (name, y_pred) in zip(axes, cls_preds):
    cm = confusion_matrix(y_test_cls, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Down', 'Up']).plot(ax=ax, cmap='Blues')
    ax.set_title(f'{name} - Confusion Matrix', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison_classification.png")

# %% [markdown]
# ## Metrics Comparison

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Regression
models = ['Random Forest', 'XGBoost', 'SVR']
colors = ['#818CF8', '#34D399', '#FBBF24']

x = np.arange(len(models))
width = 0.3

ax1.bar(x - width, [rf_reg_mae, xgb_reg_mae, svr_mae], width, label='MAE', color='#818CF8', alpha=0.8)
ax1.bar(x, [rf_reg_rmse, xgb_reg_rmse, svr_rmse], width, label='RMSE', color='#F87171', alpha=0.8)
ax1.bar(x + width, [rf_reg_r2, xgb_reg_r2, svr_r2], width, label='R2', color='#34D399', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title('Regression Metrics Comparison', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Classification
metrics = ['Accuracy', 'F1 Score']
rf_vals = [rf_cls_acc, rf_cls_f1]
xgb_vals = [xgb_cls_acc, xgb_cls_f1]
svc_vals = [svc_acc, svc_f1]

x2 = np.arange(len(metrics))
ax2.bar(x2 - width, rf_vals, width, label='Random Forest', color='#818CF8', alpha=0.8)
ax2.bar(x2, xgb_vals, width, label='XGBoost', color='#34D399', alpha=0.8)
ax2.bar(x2 + width, svc_vals, width, label='SVC', color='#FBBF24', alpha=0.8)
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics)
ax2.set_title('Classification Metrics Comparison', fontweight='bold')
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison_metrics.png")

# %% [markdown]
# ## Feature Importance (Random Forest)

# %%
importances = rf_reg.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({'Feature': available_features, 'Importance': importances}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#818CF8', alpha=0.85)
ax.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Save Models

# %%
import joblib
import json
from datetime import datetime

# Save all models
joblib.dump(rf_reg.best_estimator_, 'random_forest_regressor.pkl')
joblib.dump(rf_cls.best_estimator_, 'random_forest_classifier.pkl')
joblib.dump(xgb_reg.best_estimator_, 'xgboost_regressor.pkl')
joblib.dump(xgb_cls.best_estimator_, 'xgboost_classifier.pkl')
joblib.dump(svr.best_estimator_, 'svm_regressor.pkl')
joblib.dump(svc.best_estimator_, 'svm_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(target_scaler, 'svm_target_scaler.pkl')

print("All models saved:")
print("   random_forest_regressor.pkl")
print("   random_forest_classifier.pkl")
print("   xgboost_regressor.pkl")
print("   xgboost_classifier.pkl")
print("   svm_regressor.pkl")
print("   svm_classifier.pkl")
print("   scaler.pkl")
print("   svm_target_scaler.pkl")

# Save training log
training_log = {
    'timestamp': datetime.now().isoformat(),
    'ticker': train_ticker,
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'features': available_features,
    'regression': {
        'Random Forest': {'mae': round(rf_reg_mae, 4), 'rmse': round(rf_reg_rmse, 4), 'r2': round(rf_reg_r2, 4), 'params': rf_reg.best_params_},
        'XGBoost': {'mae': round(xgb_reg_mae, 4), 'rmse': round(xgb_reg_rmse, 4), 'r2': round(xgb_reg_r2, 4), 'params': xgb_reg.best_params_},
        'SVR': {'mae': round(svr_mae, 4), 'rmse': round(svr_rmse, 4), 'r2': round(svr_r2, 4), 'params': {k: str(v) for k, v in svr.best_params_.items()}},
    },
    'classification': {
        'Random Forest': {'accuracy': round(rf_cls_acc, 4), 'f1': round(rf_cls_f1, 4), 'params': rf_cls.best_params_},
        'XGBoost': {'accuracy': round(xgb_cls_acc, 4), 'f1': round(xgb_cls_f1, 4), 'params': xgb_cls.best_params_},
        'SVC': {'accuracy': round(svc_acc, 4), 'f1': round(svc_f1, 4), 'params': {k: str(v) for k, v in svc.best_params_.items()}},
    },
    'best_regression_model': best_reg,
    'best_classification_model': best_cls,
}

with open('training_results.json', 'w') as f:
    json.dump(training_log, f, indent=2)

print("\nTraining results saved to training_results.json")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("TRAINING COMPLETE - STOCK MARKET AI ADVISOR")
print("=" * 70)
print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Ticker: {train_ticker}")
print(f"Train/Test Split: {X_train.shape[0]}/{X_test.shape[0]} samples")
print(f"Features: {len(available_features)}")
print(f"\nBest Regression Model:     {best_reg} (RMSE: {reg_models[best_reg]:.4f})")
print(f"Best Classification Model: {best_cls} (F1: {cls_models[best_cls]:.4f})")
print(f"\nModels saved: 8 files (.pkl)")
print(f"Plots saved: 4 images (.png)")
print(f"Log saved: training_results.json")
print("\nDISCLAIMER: This project is for educational purposes only.")
print("It does NOT provide financial advice.")
print("=" * 70)

# %% [markdown]
# ## Download Models (Colab)

# %%
# Uncomment to download from Colab:
# from google.colab import files
# for f in ['random_forest_regressor.pkl', 'random_forest_classifier.pkl',
#           'xgboost_regressor.pkl', 'xgboost_classifier.pkl',
#           'svm_regressor.pkl', 'svm_classifier.pkl',
#           'scaler.pkl', 'svm_target_scaler.pkl',
#           'training_results.json']:
#     files.download(f)
