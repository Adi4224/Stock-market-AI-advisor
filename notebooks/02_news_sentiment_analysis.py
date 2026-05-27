# %% [markdown]
# # 02 — Financial News Sentiment Analysis
# VADER-based sentiment analysis of financial news headlines.

# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backend.sentiment_analysis import analyze_sentiment, analyze_news_dataframe, get_sentiment_summary, plot_sentiment_distribution
from src.backend.news_preprocessing import preprocess_news_data

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load News Data

# %%
news_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'news_data.csv')
df = pd.read_csv(news_path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()

# %% [markdown]
# ## 2. Text Preprocessing

# %%
df_clean = preprocess_news_data(df)
print("Preprocessed columns:", list(df_clean.columns))
df_clean[['headline', 'cleaned_headline']].head()

# %% [markdown]
# ## 3. Sentiment Analysis

# %%
df_analyzed = analyze_news_dataframe(df_clean, text_col='headline')
print("Sentiment columns added:", [c for c in df_analyzed.columns if 'sentiment' in c])
df_analyzed[['headline', 'sentiment_score', 'sentiment_label']].head(10)

# %% [markdown]
# ## 4. Sentiment Distribution

# %%
summary = get_sentiment_summary(df_analyzed)
print("Sentiment Summary:")
for k, v in summary.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")

plot_sentiment_distribution(df_analyzed, save_path=os.path.join(FIGURES_DIR, 'sentiment_distribution.png'))

# %% [markdown]
# ## 5. Sentiment by Ticker

# %%
if 'ticker' in df_analyzed.columns:
    ticker_sentiment = df_analyzed.groupby('ticker')['sentiment_score'].agg(['mean', 'std', 'count'])
    ticker_sentiment = ticker_sentiment.sort_values('mean', ascending=False)
    print(ticker_sentiment)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#00C853' if v >= 0 else '#FF5252' for v in ticker_sentiment['mean']]
    ax.barh(ticker_sentiment.index, ticker_sentiment['mean'], color=colors, alpha=0.8)
    ax.set_title('Average Sentiment Score by Ticker', fontsize=14)
    ax.set_xlabel('Average Compound Score')
    ax.axvline(x=0, color='white', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_by_ticker.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 6. Save Processed Data

# %%
output_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_news_data.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_analyzed.to_csv(output_path, index=False)
print(f"Processed news data saved to: {output_path}")

if __name__ == '__main__':
    print("\nSentiment analysis complete!")
