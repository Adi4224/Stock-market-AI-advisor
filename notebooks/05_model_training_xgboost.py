# %% [markdown]
# # 05 — XGBoost Model Training
# Train XGBoost Regressor and Classifier for stock prediction.

# %%
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib, seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.feature_engineering import add_technical_features, add_targets, get_feature_columns
from src.backend.stock_preprocessing import get_train_test_split, scale_features
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

if 'Ticker' in df.columns:
    df = df[df['Ticker'] == df['Ticker'].value_counts().index[0]].copy()

df = add_technical_features(df)
df = add_targets(df)
df = df.dropna().reset_index(drop=True)
feature_cols = [c for c in get_feature_columns() if c in df.columns]
print(f"Shape: {df.shape}, Features: {len(feature_cols)}")

# %% [markdown]
# ## 2. Train-Test Split

# %%
X_train, X_test, y_train_reg, y_test_reg = get_train_test_split(df, 'next_day_close', feature_cols, 0.2, True)
_, _, y_train_cls, y_test_cls = get_train_test_split(df, 'movement', feature_cols, 0.2, True)
X_train_s, X_test_s, _ = scale_features(X_train, X_test)

# %% [markdown]
# ## 3. Train XGBoost Regressor

# %%
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
tscv = TimeSeriesSplit(n_splits=3)
xgb_reg = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_reg.fit(X_train_s, y_train_reg)
print(f"Best params: {xgb_reg.best_params_}")

y_pred_reg = xgb_reg.predict(X_test_s)
reg_metrics = evaluate_regression(y_test_reg, y_pred_reg, 'XGBoost Regressor')
print(f"Regression: {reg_metrics}")
plot_actual_vs_predicted(y_test_reg, y_pred_reg, 'XGBoost', save_path=os.path.join(PLOTS_DIR, 'xgb_actual_vs_predicted.png'))

# %% [markdown]
# ## 4. Train XGBoost Classifier

# %%
param_grid_cls = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
xgb_cls = GridSearchCV(XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss'), param_grid_cls, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
xgb_cls.fit(X_train_s, y_train_cls)
y_pred_cls = xgb_cls.predict(X_test_s)
cls_metrics = evaluate_classification(y_test_cls, y_pred_cls, 'XGBoost Classifier')
print(f"Classification: {cls_metrics}")
plot_confusion_matrix(y_test_cls, y_pred_cls, 'XGBoost', save_path=os.path.join(PLOTS_DIR, 'xgb_confusion_matrix.png'))

# %% [markdown]
# ## 5. Feature Importance

# %%
importances = xgb_reg.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#00C853', alpha=0.8)
ax.set_title('XGBoost — Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'xgb_feature_importance.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Save Models

# %%
joblib.dump(xgb_reg.best_estimator_, os.path.join(MODEL_DIR, 'xgboost_regressor.pkl'))
joblib.dump(xgb_cls.best_estimator_, os.path.join(MODEL_DIR, 'xgboost_classifier.pkl'))
with open(os.path.join(LOGS_DIR, 'xgb_training_log.json'), 'w') as f:
    json.dump({'reg': reg_metrics, 'cls': cls_metrics, 'params_reg': xgb_reg.best_params_, 'params_cls': xgb_cls.best_params_}, f, indent=2)
print("XGBoost models saved!")

if __name__ == '__main__':
    print("\nXGBoost training complete!")
