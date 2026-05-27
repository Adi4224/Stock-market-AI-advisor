# %% [markdown]
# # 03 — Customer Analytics & User Segmentation
# KMeans clustering for investor profiling.

# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.customer_segmentation import (
    compute_risk_score, classify_investor_type,
    train_kmeans_segmentation, segment_user, plot_user_segments
)
from src.backend.user_preprocessing import preprocess_user_data

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load User Data

# %%
user_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'user_behavior.csv')
df = pd.read_csv(user_path)
print(f"Shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Preprocess

# %%
df_processed = preprocess_user_data(df)
print(f"Processed shape: {df_processed.shape}")
print(f"New columns: {[c for c in df_processed.columns if c not in df.columns]}")

# %% [markdown]
# ## 3. Risk Score Computation

# %%
risk_scores = []
segments = []
for _, row in df.iterrows():
    user_data = row.to_dict()
    score = compute_risk_score(user_data)
    segment = classify_investor_type(score)
    risk_scores.append(score)
    segments.append(segment)

df['risk_score'] = risk_scores
df['investor_type'] = segments
print(df['investor_type'].value_counts())

# %% [markdown]
# ## 4. KMeans Clustering

# %%
kmeans, label_map, scaler = train_kmeans_segmentation(df, n_clusters=3)
print(f"Label map: {label_map}")

# %% [markdown]
# ## 5. Visualization

# %%
plot_user_segments(df, df['investor_type'].values,
                   save_path=os.path.join(FIGURES_DIR, 'user_segments.png'))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df['risk_score'].hist(bins=20, ax=axes[0], color='#00D4FF', alpha=0.7)
axes[0].set_title('Risk Score Distribution')
df.groupby('investor_type')['risk_tolerance'].mean().plot(kind='bar', ax=axes[1], color=['#00D4FF','#FFD700','#FF5252'])
axes[1].set_title('Avg Risk Tolerance by Type')
df['investor_type'].value_counts().plot(kind='pie', ax=axes[2], colors=['#00D4FF','#FFD700','#FF5252'], autopct='%1.1f%%')
axes[2].set_title('Investor Type Distribution')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'customer_analytics.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Sample Segmentation

# %%
test_users = [
    {'age': 60, 'risk_tolerance': 2, 'trading_frequency': 'quarterly', 'past_returns': 5},
    {'age': 35, 'risk_tolerance': 5, 'trading_frequency': 'weekly', 'past_returns': 12},
    {'age': 25, 'risk_tolerance': 9, 'trading_frequency': 'daily', 'past_returns': 30},
]
for u in test_users:
    result = segment_user(u)
    print(f"Age {u['age']}, Risk {u['risk_tolerance']} -> {result['segment']} (Score: {result['risk_score']})")

# %%
output_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_user_data.csv')
df.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")

if __name__ == '__main__':
    print("\nCustomer analytics complete!")
