# %% [markdown]
# # 04 — Random Forest Model Training
# Train Random Forest Regressor and Classifier for stock prediction.

# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib, pickle, seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, add_targets, get_feature_columns
from src.backend.stock_preprocessing import preprocess_stock_data, get_train_test_split, scale_features
from src.backend.evaluate_models import evaluate_regression, evaluate_classification, plot_confusion_matrix, plot_actual_vs_predicted

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_plots')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'training_logs')
for d in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]: os.makedirs(d, exist_ok=True)

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

print(f"Raw data shape: {df.shape}")

# %% [markdown]
# ## 2. Feature Engineering

# %%
if 'Ticker' in df.columns:
    df = df[df['Ticker'] == df['Ticker'].value_counts().index[0]].copy()

df = add_technical_features(df)
df = add_targets(df)
df = df.dropna().reset_index(drop=True)
feature_cols = get_feature_columns()
available = [c for c in feature_cols if c in df.columns]
print(f"Features: {available}")
print(f"Clean data shape: {df.shape}")

# %% [markdown]
# ## 3. Train-Test Split

# %%
X_train, X_test, y_train_reg, y_test_reg = get_train_test_split(df, 'next_day_close', available, test_size=0.2, time_based=True)
_, _, y_train_cls, y_test_cls = get_train_test_split(df, 'movement', available, test_size=0.2, time_based=True)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, os.path.join(MODEL_DIR, 'scaler.pkl'))
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 4. Train Random Forest Regressor

# %%
param_grid_reg = {'n_estimators': [100, 200], 'max_depth': [10, 15, None], 'min_samples_split': [2, 5]}
tscv = TimeSeriesSplit(n_splits=3)
rf_reg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_reg, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_reg.fit(X_train_scaled, y_train_reg)
print(f"Best params: {rf_reg.best_params_}")

y_pred_reg = rf_reg.predict(X_test_scaled)
reg_metrics = evaluate_regression(y_test_reg, y_pred_reg, 'Random Forest Regressor')
print(f"Regression: {reg_metrics}")

plot_actual_vs_predicted(y_test_reg, y_pred_reg, 'Random Forest', save_path=os.path.join(PLOTS_DIR, 'rf_actual_vs_predicted.png'))

# %% [markdown]
# ## 5. Train Random Forest Classifier

# %%
param_grid_cls = {'n_estimators': [100, 200], 'max_depth': [10, 15, None], 'min_samples_split': [2, 5]}
rf_cls = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_cls, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
rf_cls.fit(X_train_scaled, y_train_cls)
print(f"Best params: {rf_cls.best_params_}")

y_pred_cls = rf_cls.predict(X_test_scaled)
cls_metrics = evaluate_classification(y_test_cls, y_pred_cls, 'Random Forest Classifier')
print(f"Classification: {cls_metrics}")

plot_confusion_matrix(y_test_cls, y_pred_cls, 'Random Forest', save_path=os.path.join(PLOTS_DIR, 'rf_confusion_matrix.png'))

# %% [markdown]
# ## 6. Feature Importance

# %%
importances = rf_reg.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({'Feature': available, 'Importance': importances}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#00D4FF', alpha=0.8)
ax.set_title('Random Forest — Feature Importance', fontsize=14)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rf_feature_importance.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Save Models

# %%
joblib.dump(rf_reg.best_estimator_, os.path.join(MODEL_DIR, 'random_forest_regressor.pkl'))
joblib.dump(rf_cls.best_estimator_, os.path.join(MODEL_DIR, 'random_forest_classifier.pkl'))
print("Models saved!")

# Save training log
import json
log = {'reg_metrics': reg_metrics, 'cls_metrics': cls_metrics, 'best_params_reg': rf_reg.best_params_, 'best_params_cls': rf_cls.best_params_}
with open(os.path.join(LOGS_DIR, 'rf_training_log.json'), 'w') as f:
    json.dump(log, f, indent=2)

if __name__ == '__main__':
    print("\nRandom Forest training complete!")
