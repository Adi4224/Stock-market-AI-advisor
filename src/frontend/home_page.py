"""Home Page — Landing page with project overview and feature highlights."""

import streamlit as st


def render():
    C = st.session_state.get("_colors", {})
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="animate-fade-in" style="text-align:center; padding: 2.5rem 0 1rem;">
            <h1 style="font-size:2.8rem; margin-bottom:0;">
                <span class="shimmer">AI-Powered Stock Market Advisor</span>
            </h1>
            <p class="muted" style="font-size:1.15rem; margin-top:8px;">
                Predict &bull; Analyze &bull; Invest Smarter
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="disclaimer-banner" style="text-align:center; margin:0 auto 2rem; max-width:700px;">
            ⚠️ <strong>This application is for educational purposes only</strong> and does not provide financial advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature Cards ────────────────────────────────────────────────────────
    st.markdown("### ✨ Key Features")

    features = [
        ("🎯", "Stock Prediction", "Predict next-day closing prices and movement using Random Forest, XGBoost, and LSTM.", accent),
        ("💬", "Sentiment Analysis", "Analyze financial news sentiment using VADER NLP to classify headlines.", green),
        ("👤", "Customer Analytics", "Profile investors by risk tolerance and suggest strategies using KMeans.", gold),
        ("💡", "Smart Suggestions", "Get stock recommendations matched to your risk profile and goals.", red),
    ]

    cols = st.columns(4)
    for col, (icon, title, desc, color) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; min-height:250px; border-top: 3px solid {color};">
                    <div style="font-size:2.5rem; margin-bottom:10px;">{icon}</div>
                    <h4 style="color:{color}; margin:0 0 8px;">{title}</h4>
                    <p class="muted" style="font-size:0.85rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Stats ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Project at a Glance")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Stocks Tracked", "50")
    s2.metric("ML Models", "7")
    s3.metric("Features Engineered", "16")
    s4.metric("NLP Engine", "VADER")
    s5.metric("User Segments", "3")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Overview + Tech Stack ────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 📋 Project Overview")
        st.markdown(
            """
            This **Machine Learning internship assessment project** combines
            multiple AI/ML techniques for comprehensive stock market analysis:

            - **Regression** — Predict next-day stock closing prices
            - **Classification** — Predict stock price movement (Up / Down)
            - **NLP** — Analyze financial news sentiment
            - **Customer Analytics** — Segment investors by risk profile
            - **Recommendation Engine** — Suggest stocks based on user profiles

            The system uses historical data from the
            [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
            updated through 2026 using the **yfinance** API.
            """
        )

    with right:
        st.markdown("### 🛠️ Technology Stack")
        techs = [
            ("Python", "badge-accent"), ("Pandas", "badge-accent"), ("NumPy", "badge-accent"),
            ("scikit-learn", "badge-green"), ("XGBoost", "badge-green"),
            ("TensorFlow / Keras", "badge-green"), ("LSTM", "badge-green"),
            ("VADER / NLTK", "badge-gold"), ("yfinance", "badge-gold"),
            ("Streamlit", "badge-red"), ("Plotly", "badge-red"),
            ("Matplotlib / Seaborn", "badge-red"),
        ]
        badges_html = " ".join(
            f'<span class="badge {cls}" style="margin:3px;">{name}</span>'
            for name, cls in techs
        )
        st.markdown(f'<div style="line-height:2.2;">{badges_html}</div>', unsafe_allow_html=True)

    # ── Dataset Info ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📂 Dataset Information")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="color: {accent};">📥 Base Dataset</h4>
                <p class="muted">
                    <strong>Source:</strong> Kaggle — jacksoncrow/stock-market-dataset<br>
                    <strong>Contents:</strong> Historical OHLCV data for NYSE/NASDAQ tickers<br>
                    <strong>Format:</strong> Individual CSV per ticker symbol
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="color: {green};">🔄 Data Update</h4>
                <p class="muted">
                    <strong>Method:</strong> yfinance API<br>
                    <strong>Process:</strong> Detect last date → Fetch missing → Merge → Deduplicate<br>
                    <strong>Coverage:</strong> Extended through May 2026
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p class="muted" style="text-align:center;">Use the dropdown menu above to explore the full application.</p>',
        unsafe_allow_html=True,
    )
