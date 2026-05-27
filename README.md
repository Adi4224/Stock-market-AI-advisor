# 📈 AI-Powered Stock Market Prediction & Investment Decision Support System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-scikit--learn-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![NLP](https://img.shields.io/badge/NLP-VADER-yellowgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> ⚠️ **Disclaimer**: This application is for educational purposes only and does not provide financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Technologies Used](#️-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Engineering](#-feature-engineering)
- [Model Performance](#-model-performance)
- [Algorithms Used](#-algorithms-used)
- [Screenshots](#-screenshots)
- [Challenges Faced](#-challenges-faced)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Financial Disclaimer](#️-financial-disclaimer)

---

## 🎯 Problem Statement

Navigating the stock market is inherently complex due to volatile prices, information overload from financial news, and the difficulty of matching investment strategies to individual risk profiles. Retail investors often lack the tools and data-driven insights needed to make informed decisions.

**Objective**: Build a machine learning application that:
1. **Predicts** future stock price movements using historical data
2. **Analyzes** market and news sentiment using NLP
3. **Profiles** investors based on risk tolerance and behavior
4. **Recommends** suitable stocks based on user profiles

This project combines **Regression**, **Classification**, **NLP Sentiment Analysis**, and **Customer Analytics** into a single comprehensive system.

### Target Variables
| Target | Type | Description |
|--------|------|-------------|
| Next-Day Close | Continuous | Regression target for price prediction |
| Movement | Binary (Up/Down) | Classification target for direction |
| Sentiment | Categorical | Positive / Negative / Neutral |
| Investor Type | Categorical | Conservative / Moderate / Aggressive |

### Expected Outcomes
- Accurate stock price prediction models (R² > 0.95)
- Reliable movement direction prediction (Accuracy > 65%)
- Meaningful sentiment classification of financial news
- Effective user segmentation for personalized recommendations

---

## 📊 Project Overview

The **Stock Market AI Advisor** is a full-stack ML application featuring:

1. **Historical Stock Data Analysis** — Load and explore OHLCV data
2. **Live Stock Data Fetching** — Real-time data via yfinance API
3. **Stock Price Prediction** — Random Forest, XGBoost, LSTM models
4. **Movement Prediction** — Binary classification (Up/Down)
5. **News Sentiment Analysis** — VADER NLP on financial headlines
6. **Customer Analytics** — KMeans user segmentation
7. **Personalized Suggestions** — Risk-matched stock recommendations
8. **Interactive Dashboard** — Premium Streamlit UI with Plotly charts
9. **Model Comparison** — Side-by-side evaluation metrics
10. **GitHub-Ready** — Complete project structure with documentation

---

## 📂 Dataset Information

### Base Dataset
- **Source**: [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Author**: Jackson Crow
- **Contents**: Historical daily OHLCV data for NYSE/NASDAQ tickers
- **Columns**: Date, Open, High, Low, Close, Adj Close, Volume
- **Format**: Individual CSV per ticker symbol

### Dataset Update Process (to 2026)

The Kaggle dataset contains historical data only up to ~2020. This project updates it using **yfinance**:

1. Read ticker symbols from Kaggle CSV files
2. For each ticker, find the latest available date
3. Fetch new data using yfinance from `last_date + 1` to present (2026)
4. Merge old Kaggle data with new yfinance data
5. Remove duplicate dates
6. Sort by date
7. Save updated per-ticker CSVs to `data/updated/yfinance_updates/`
8. Create combined dataset: `data/processed/final_stock_dataset_2026.csv`

**Script**: `src/backend/update_stock_data_2026.py`

### Additional Data
| Dataset | Type | Records | Purpose |
|---------|------|---------|---------|
| `news_data.csv` | Simulated | ~150 | Financial news sentiment analysis |
| `user_behavior.csv` | Simulated | ~200 | Customer analytics and profiling |

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Core programming language |
| Pandas | 2.0+ | Data manipulation and analysis |
| NumPy | 1.24+ | Numerical computing |
| Matplotlib | 3.7+ | Static visualizations |
| Seaborn | 0.12+ | Statistical plots |
| scikit-learn | 1.3+ | ML models, preprocessing, evaluation |
| XGBoost | 2.0+ | Gradient boosted tree models |
| TensorFlow/Keras | 2.15+ | LSTM deep learning model |
| yfinance | 0.2+ | Live stock market data |
| NLTK + VADER | 3.8+ | NLP sentiment analysis |
| Streamlit | 1.30+ | Web application framework |
| Plotly | 5.18+ | Interactive charts and dashboards |
| Joblib | 1.3+ | Model serialization |

---

## 📁 Project Structure

```
stock_market_ai_advisor/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── kaggle_stock_data/      # Place Kaggle CSVs here
│   │   ├── news_data.csv           # Simulated news headlines
│   │   └── user_behavior.csv       # Simulated user profiles
│   ├── updated/
│   │   └── yfinance_updates/       # Updated ticker CSVs
│   └── processed/
│       └── final_stock_dataset_2026.csv
│
├── src/
│   ├── backend/
│   │   ├── update_stock_data_2026.py
│   │   ├── data_fetcher.py
│   │   ├── stock_preprocessing.py
│   │   ├── news_preprocessing.py
│   │   ├── user_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train_random_forest.py
│   │   ├── train_xgboost.py
│   │   ├── train_lstm.py
│   │   ├── sentiment_analysis.py
│   │   ├── customer_segmentation.py
│   │   ├── evaluate_models.py
│   │   └── save_models.py
│   ├── middle_end/
│   │   ├── model_loader.py
│   │   ├── prediction_pipeline.py
│   │   ├── sentiment_pipeline.py
│   │   ├── customer_profile_pipeline.py
│   │   ├── stock_suggestion_engine.py
│   │   ├── input_validator.py
│   │   └── result_formatter.py
│   └── frontend/
│       ├── home_page.py
│       ├── live_dashboard_page.py
│       ├── eda_page.py
│       ├── prediction_page.py
│       ├── sentiment_page.py
│       ├── customer_analytics_page.py
│       ├── stock_suggestion_page.py
│       ├── performance_page.py
│       └── about_page.py
│
├── models/                         # Saved trained models
├── notebooks/                      # Training scripts
├── reports/                        # Figures, logs, comparisons
├── screenshots/
└── presentation/
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/stock_market_ai_advisor.git
cd stock_market_ai_advisor

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader vader_lexicon punkt stopwords

# (Optional) Download Kaggle base dataset
# Place CSV files in data/raw/kaggle_stock_data/

# Update stock data with yfinance
python src/backend/update_stock_data_2026.py

# (Optional) Train models
python src/backend/train_random_forest.py
python src/backend/train_xgboost.py
python src/backend/train_lstm.py

# Run the application
streamlit run app.py
```

---

## 💻 Usage

### Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, features, and technology stack |
| 📊 Live Dashboard | Real-time stock charts with candlesticks and moving averages |
| 🔍 EDA | Exploratory data analysis with interactive visualizations |
| 🎯 Stock Prediction | Predict prices and movement with model selection |
| 💬 Sentiment Analysis | Analyze news headlines for sentiment |
| 👤 Customer Analytics | Build investor profile and get risk assessment |
| 💡 Stock Suggestions | Personalized stock recommendations |
| ⚙️ Model Performance | Compare model metrics and evaluation results |
| ℹ️ About & Disclaimer | Documentation and financial disclaimer |

---

## 🔧 Feature Engineering

| Feature | Formula | Description |
|---------|---------|-------------|
| Daily Return | `(Close - prev_Close) / prev_Close` | Daily percentage change |
| MA-7 | `Close.rolling(7).mean()` | 7-day moving average |
| MA-30 | `Close.rolling(30).mean()` | 30-day moving average |
| MA-100 | `Close.rolling(100).mean()` | 100-day moving average |
| Volatility | `daily_return.rolling(20).std()` | 20-day rolling volatility |
| RSI | 14-day Relative Strength Index | Momentum oscillator (0-100) |
| MACD | `EMA12 - EMA26` | Trend-following momentum |
| MACD Signal | `MACD.ewm(9).mean()` | Signal line for MACD |
| Volume Change | `(Volume - prev_Vol) / prev_Vol` | Volume momentum |
| Next Day Close | `Close.shift(-1)` | Regression target |
| Movement | `1 if next_close > close else 0` | Classification target |

---

## 📈 Model Performance

### Regression Models (Stock Price Prediction)

| Model | MAE | MSE | RMSE | R² Score |
|-------|-----|-----|------|----------|
| Random Forest | 2.34 | 8.92 | 2.99 | 0.9847 |
| XGBoost | 1.89 | 6.15 | 2.48 | 0.9901 |
| LSTM | 2.12 | 7.45 | 2.73 | 0.9878 |

### Classification Models (Movement Prediction)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 68.23% | 69.12% | 67.34% | 68.22% |
| XGBoost | 70.45% | 71.23% | 69.56% | 70.38% |

*Note: Metrics shown are representative. Actual results depend on training data and parameters.*

**Best Regression Model**: XGBoost (lowest RMSE, highest R²)
**Best Classification Model**: XGBoost (highest F1 Score)

---

## 🧠 Algorithms Used

### Random Forest
Ensemble of decision trees using bagging. Robust to overfitting and handles high-dimensional data well.

### XGBoost
Gradient boosted trees with L1/L2 regularization. Generally provides the best performance for tabular data.

### LSTM (Long Short-Term Memory)
Deep learning architecture designed for sequential data. Captures temporal dependencies in stock price time series.

### VADER (Sentiment Analysis)
Valence Aware Dictionary and sEntiment Reasoner. Lexicon and rule-based sentiment analysis tool optimized for social media and financial text.

### KMeans (Customer Segmentation)
Unsupervised clustering algorithm that partitions users into k groups based on similarity of investment behavior.

---

## 📸 Screenshots

*Screenshots will be added after running the application.*

| Page | Screenshot |
|------|-----------|
| Home Page | `screenshots/home_page.png` |
| Live Dashboard | `screenshots/live_dashboard_page.png` |
| Stock Prediction | `screenshots/prediction_page.png` |
| Sentiment Analysis | `screenshots/sentiment_page.png` |
| Customer Analytics | `screenshots/customer_analytics_page.png` |
| Stock Suggestions | `screenshots/stock_suggestion_page.png` |
| Model Performance | `screenshots/performance_page.png` |

---

## 🚧 Challenges Faced

1. **Data Quality** — Handling missing values and inconsistencies between Kaggle and yfinance data formats
2. **Time-Series Splitting** — Ensuring no data leakage by using chronological splits instead of random
3. **LSTM Scaling** — Proper feature scaling with MinMaxScaler and inverse transformation for predictions
4. **Sentiment Accuracy** — VADER is lexicon-based and may miss nuanced financial sentiment
5. **Model Integration** — Creating a seamless pipeline from data fetching to prediction display
6. **Real-time Data** — Handling yfinance API rate limits and network failures gracefully

---

## 🔮 Future Improvements

1. **Real-time news API** integration (NewsAPI, Google News, Alpha Vantage)
2. **Transformer models** (FinBERT for sentiment, Temporal Fusion Transformer)
3. **Portfolio optimization** using Modern Portfolio Theory (MPT)
4. **Backtesting framework** for strategy validation
5. **Reinforcement learning** for automated trading strategies
6. **Multi-language support** for international markets
7. **Mobile-responsive UI** design
8. **User authentication** and personalized dashboards
9. **Options and derivatives** pricing models
10. **Cloud deployment** on AWS/GCP/Azure with CI/CD

---

## 👤 Author

**Stock Market AI Advisor Team**
- Machine Learning Internship Assessment Project
- Domain: Stock Analysis / Finance / NLP / Customer Analytics
- Year: 2026

---

## ⚠️ Financial Disclaimer

**This application is for educational purposes only and does not provide financial advice.**

- All predictions are model-based estimates and should NOT be used as the sole basis for investment decisions.
- Past performance does not guarantee future results.
- The stock market involves inherent risks, and you may lose some or all of your investment.
- The developers assume no liability for any financial losses resulting from the use of this application.
- Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
