"""About & Disclaimer Page — Project documentation and financial disclaimer."""

import streamlit as st


def render():
    C = st.session_state.get("_colors", {})
    accent = C.get("accent", "#818CF8")
    red = C.get("red", "#F87171")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">ℹ️ About & Disclaimer</h1>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 About", "📊 Dataset", "🛠️ Architecture", "⚠️ Disclaimer", "👤 Author"])

    with tab1:
        _render_about(accent)
    with tab2:
        _render_dataset(accent)
    with tab3:
        _render_architecture()
    with tab4:
        _render_disclaimer(red)
    with tab5:
        _render_author(accent)


def _render_about(accent):
    st.markdown('Developed by **Adithya Dadi** as part of the Summer Internship program for Agentic AI at **DataPro**.')
    st.markdown("""
    ### 📋 Project Overview

    The **AI-Powered Stock Market Prediction & Investment Decision Support System** combines
    multiple AI/ML techniques for comprehensive stock market analysis.

    ### 🎯 Problem Statement

    Navigating the stock market is complex due to volatile prices, information overload, and
    difficulty matching strategies to individual risk profiles. This project addresses these through:

    1. **Stock Price Prediction** — ML models forecast next-day closing prices
    2. **Movement Classification** — Predict price direction (Up/Down)
    3. **Sentiment Analysis** — VADER NLP on financial news
    4. **Customer Analytics** — Segment investors by risk profile

    ### 🧠 Algorithms Used

    | Algorithm | Task | Description |
    |-----------|------|-------------|
    | Random Forest | Regression & Classification | Ensemble of decision trees with bagging |
    | XGBoost | Regression & Classification | Gradient boosted trees with regularization |
    | LSTM | Time-Series Regression | Deep learning for sequential data patterns |
    | VADER | NLP Sentiment Analysis | Lexicon-based sentiment scoring |
    | KMeans | Customer Segmentation | Unsupervised clustering for user profiling |

    ### 🔧 Feature Engineering

    | Feature | Description |
    |---------|-------------|
    | Daily Return | Percentage change in closing price |
    | MA-7, MA-14, MA-30, MA-50 | Moving averages for trend identification |
    | Price Range | Daily high minus low price spread |
    | Volatility | 20-day rolling standard deviation |
    | RSI | Relative Strength Index (14-day) |
    | MACD | Moving Average Convergence Divergence |
    | Volume Change | Percentage change in trading volume |
    """)


def _render_dataset(accent):
    st.markdown("""
    ### 📊 Dataset Information

    #### Base Dataset
    - **Source**: [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
    - **Contents**: Historical daily OHLCV data for NYSE/NASDAQ tickers
    - **Format**: Individual CSV per ticker symbol

    #### Update Process (to 2026)
    1. Load Kaggle CSVs → Detect last date → Fetch via yfinance → Merge → Deduplicate → Save

    #### Additional Data
    | Dataset | Records | Purpose |
    |---------|---------|---------|
    | `news_data.csv` | ~150 | Financial news sentiment analysis |
    | `user_behavior.csv` | ~200 | Customer analytics and profiling |
    """)


def _render_architecture():
    st.markdown("""
    ### 🛠️ System Architecture

    ```
    ┌────────────────────────────────────────────────────────────┐
    │                    STREAMLIT FRONTEND                       │
    │  Dashboard | EDA | Prediction | Sentiment | Analytics      │
    └──────────────────────────┬─────────────────────────────────┘
                               │
    ┌──────────────────────────▼─────────────────────────────────┐
    │                    MIDDLE-END LAYER                         │
    │  Model Loader | Prediction | Sentiment | Customer Pipeline │
    └──────────────────────────┬─────────────────────────────────┘
                               │
    ┌──────────────────────────▼─────────────────────────────────┐
    │                    BACKEND LAYER                            │
    │  Data Fetcher | Feature Engineering | Model Training        │
    │  Sentiment Analysis | Customer Segmentation                 │
    └──────────────────────────┬─────────────────────────────────┘
                               │
    ┌──────────────────────────▼─────────────────────────────────┐
    │                    DATA LAYER                               │
    │  Kaggle Dataset | yfinance API | News Data | User Data     │
    └────────────────────────────────────────────────────────────┘
    ```

    ### 🛠️ Technology Stack

    | Technology | Purpose |
    |-----------|---------|
    | Python 3.9+ | Core language |
    | Pandas / NumPy | Data manipulation |
    | scikit-learn | ML models and preprocessing |
    | XGBoost | Gradient boosted trees |
    | TensorFlow/Keras | LSTM deep learning |
    | NLTK + VADER | Sentiment analysis |
    | yfinance | Live stock data |
    | Streamlit + Plotly | Web dashboard and charts |
    """)


def _render_disclaimer(red):
    st.markdown(
        f'<div style="text-align:center; padding:2rem 0;">'
        f'<div style="font-size:3rem;">⚠️</div>'
        f'<h2 style="color:{red};">Financial Disclaimer</h2></div>',
        unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="glass-card" style="border: 2px solid rgba({_hex_to_rgb(red)},0.25); max-width:800px; margin:0 auto;">

        ### ⚠️ Important Notice

        **This application is for educational purposes only and does not provide financial advice.**

        - All predictions are **model-based estimates** and should NOT be used as the sole basis for investment decisions.
        - **Past performance does not guarantee future results.**
        - The stock market involves inherent risks, and you may lose some or all of your investment.
        - This project was created as a **Machine Learning internship assessment** by **Adithya Dadi - DataPro Summer Internship**.
        - The developers assume **no liability** for any financial losses.
        - Always consult a **qualified financial advisor** before making investment decisions.

        ### 📚 Educational Purpose

        This project demonstrates: Regression & Classification, NLP Sentiment Analysis,
        Unsupervised Learning, Full-stack development, and Software engineering best practices.

        ### 📄 License — MIT License

        </div>
        """,
        unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center; padding:2rem 0;">'
        '<p class="muted">© 2026 Adithya Dadi | Summer Internship - Agentic AI | DataPro</p></div>',
        unsafe_allow_html=True)


def _render_author(accent):
    st.markdown(
        f'<div style="text-align:center; padding:1.5rem 0;">'
        f'<div style="font-size:3rem;">👤</div>'
        f'<h2 style="color:{accent};">Project Author</h2></div>',
        unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="glass-card" style="max-width:800px; margin:0 auto;">

        | | |
        |---|---|
        | **Author** | Adithya Dadi |
        | **Project** | AI-Powered Stock Market Prediction & Investment Decision Support System |
        | **Organization** | DataPro |
        | **Program** | Summer Internship — Agentic AI |
        | **Year** | 2026 |

        ---

        ### 🚀 About the Project

        This system leverages **ensemble machine learning**, **deep learning (LSTM)**,
        **NLP-driven sentiment analysis**, and **unsupervised customer segmentation** to deliver
        a comprehensive, end-to-end stock market prediction and investment decision support platform.
        It is designed as a practical demonstration of agentic AI principles applied to the
        financial domain.

        ---

        ### 🙏 Acknowledgments

        Special thanks to the **DataPro** team and mentors for their guidance and support
        throughout the Summer Internship program. This project was made possible by the
        resources, mentorship, and collaborative environment provided by the organization.

        </div>
        """,
        unsafe_allow_html=True)


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
