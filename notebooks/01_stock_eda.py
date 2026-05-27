# %% [markdown]
# # 01 — Stock Market Exploratory Data Analysis
# 
# Comprehensive EDA on stock market data including price trends,
# volume analysis, moving averages, correlations, and distributions.

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Dataset

# %%
data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path}")
else:
    print("Processed dataset not found. Fetching sample data...")
    from src.backend.data_fetcher import fetch_live_stock_data
    frames = []
    for ticker in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']:
        d = fetch_live_stock_data(ticker, '5y')
        if d is not None:
            d = d.reset_index()
            d['Ticker'] = ticker
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# %% [markdown]
# ## 2. Dataset Overview

# %%
print("--- Data Types ---")
print(df.dtypes)
print("\n--- Statistical Summary ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(f"Duplicates: {df.duplicated().sum()}")

if 'Ticker' in df.columns:
    print(f"\nTickers: {df['Ticker'].nunique()}")
    print(df['Ticker'].value_counts())

# %%
# Sample records
df.head(10)

# %% [markdown]
# ## 3. Stock Price Trends

# %%
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

if 'Ticker' in df.columns:
    top_tickers = df['Ticker'].value_counts().head(5).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 6))
    for ticker in top_tickers:
        t_data = df[df['Ticker'] == ticker].sort_values('Date')
        ax.plot(t_data['Date'], t_data['Close'], label=ticker, linewidth=1.2)
    ax.set_title('Stock Price Trends (Top 5 Tickers)', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'stock_price_trends.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 4. Volume Analysis

# %%
if 'Ticker' in df.columns:
    fig, ax = plt.subplots(figsize=(14, 6))
    for ticker in top_tickers[:3]:
        t_data = df[df['Ticker'] == ticker].sort_values('Date')
        ax.plot(t_data['Date'], t_data['Volume'], label=ticker, alpha=0.7)
    ax.set_title('Trading Volume Trends', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'volume_trends.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 5. Moving Averages

# %%
if 'Ticker' in df.columns:
    ticker = top_tickers[0]
    t_data = df[df['Ticker'] == ticker].sort_values('Date').copy()
    t_data['MA_7'] = t_data['Close'].rolling(7).mean()
    t_data['MA_30'] = t_data['Close'].rolling(30).mean()
    t_data['MA_100'] = t_data['Close'].rolling(100).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_data['Date'], t_data['Close'], label='Close', linewidth=1.5, color='white')
    ax.plot(t_data['Date'], t_data['MA_7'], label='MA-7', linewidth=1, color='#00D4FF')
    ax.plot(t_data['Date'], t_data['MA_30'], label='MA-30', linewidth=1, color='#FFD700')
    ax.plot(t_data['Date'], t_data['MA_100'], label='MA-100', linewidth=1, color='#FF5252')
    ax.set_title(f'{ticker} — Moving Averages', fontsize=16)
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'moving_averages.png'), dpi=150, facecolor='#0E1117')
    plt.show()

# %% [markdown]
# ## 6. Correlation Heatmap

# %%
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
available = [c for c in numeric_cols if c in df.columns]
corr = df[available].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Daily Return Distribution

# %%
if 'Ticker' in df.columns:
    ticker = top_tickers[0]
    t_data = df[df['Ticker'] == ticker].sort_values('Date').copy()
    t_data['daily_return'] = t_data['Close'].pct_change() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    t_data['daily_return'].dropna().hist(bins=60, ax=ax, color='#00D4FF', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f'{ticker} — Daily Return Distribution', fontsize=14)
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'daily_return_distribution.png'), dpi=150)
    plt.show()

    print(f"Mean: {t_data['daily_return'].mean():.4f}%")
    print(f"Std:  {t_data['daily_return'].std():.4f}%")
    print(f"Skew: {t_data['daily_return'].skew():.4f}")
    print(f"Kurt: {t_data['daily_return'].kurtosis():.4f}")

# %% [markdown]
# ## 8. Volatility Analysis

# %%
if 'Ticker' in df.columns:
    t_data['volatility'] = t_data['daily_return'].rolling(20).std()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_data['Date'], t_data['volatility'], color='#FF5252', linewidth=1)
    ax.fill_between(t_data['Date'], t_data['volatility'], alpha=0.2, color='#FF5252')
    ax.set_title(f'{ticker} — 20-Day Rolling Volatility', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'volatility_analysis.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 9. Key Insights
# 
# 1. Stock prices show varying trends across tickers reflecting market positions
# 2. OHLC prices are highly correlated (>0.99) as expected
# 3. Daily returns follow approximately normal distribution with fat tails
# 4. Volume shows periodic spikes around earnings and major events
# 5. Volatility fluctuates over time with notable increases during uncertainty
# 6. Moving averages provide useful trend and support/resistance indicators

if __name__ == '__main__':
    print("\nEDA complete! Figures saved to:", FIGURES_DIR)
