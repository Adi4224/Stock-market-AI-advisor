"""EDA Page — Exploratory data analysis with interactive visualizations."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">🔍 Exploratory Data Analysis</h1>', unsafe_allow_html=True)

    df = _load_data()
    if df is None or df.empty:
        st.warning("No processed dataset found. Data will be fetched live.")
        return

    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0
    if rate != 1.0:
        df = df.copy()
        price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
        for c in price_cols:
            df[c] *= rate


    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview", "📈 Trends", "📊 Distribution", "🔗 Correlation", "💡 Insights"])

    with tab1:
        _render_overview(df, tmpl)
    with tab2:
        _render_trends(df, tmpl, accent, green, gold, red)
    with tab3:
        _render_distributions(df, tmpl, accent, red)
    with tab4:
        _render_correlation(df, tmpl)
    with tab5:
        _render_insights()


def _render_overview(df, tmpl):
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", str(len(df.columns)))
    c3.metric("Tickers", str(df['Ticker'].nunique()) if 'Ticker' in df.columns else "N/A")
    date_range = ""
    if 'Date' in df.columns:
        date_range = f"{df['Date'].min()} → {df['Date'].max()}"
    c4.metric("Date Range", date_range[:21] if date_range else "N/A")

    st.markdown("#### Sample Data")
    st.dataframe(df.head(10), width="stretch", height=300)

    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe().round(2), width="stretch")

    missing = df.isna().sum()
    if missing.sum() > 0:
        st.markdown("#### Missing Value Analysis")
        fig = px.bar(x=missing.index, y=missing.values, labels={'x': 'Column', 'y': 'Missing Count'},
                     color=missing.values, color_continuous_scale='Reds', template=tmpl)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, width="stretch")
    else:
        st.success("✅ No missing values found!")


def _render_trends(df, tmpl, accent, green, gold, red):
    if 'Ticker' not in df.columns:
        st.info("Ticker column not found.")
        return

    tickers = sorted(df['Ticker'].unique())
    selected = st.multiselect("Select Tickers for Trend Analysis", tickers, default=tickers[:3], key="eda_trend_tickers")
    if not selected:
        return

    sub = df[df['Ticker'].isin(selected)].copy()
    if 'Date' in sub.columns:
        sub['Date'] = pd.to_datetime(sub['Date'], utc=True)
        sub = sub.sort_values('Date')

    palette = [accent, green, gold, red, '#A78BFA']

    st.markdown("#### Stock Price Trends")
    fig = px.line(sub, x='Date', y='Close', color='Ticker', template=tmpl, color_discrete_sequence=palette)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Volume Trends")
    fig2 = px.line(sub, x='Date', y='Volume', color='Ticker', template=tmpl, color_discrete_sequence=palette)
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig2, width="stretch")

    if len(selected) == 1:
        t_data = sub.copy()
        st.markdown(f"#### Moving Averages — {selected[0]}")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_data['Date'], y=t_data['Close'], name='Close', line=dict(color=accent, width=2)))
        for ma, color in [(7, green), (30, gold), (100, red)]:
            if len(t_data) >= ma:
                fig3.add_trace(go.Scatter(x=t_data['Date'], y=t_data['Close'].rolling(ma).mean(),
                                          name=f'MA-{ma}', line=dict(color=color, width=1.2)))
        fig3.update_layout(template=tmpl, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig3, width="stretch")


def _render_distributions(df, tmpl, accent, red):
    if 'Close' in df.columns and 'Ticker' in df.columns:
        st.markdown("#### Daily Return Distribution")
        ticker_sel = st.selectbox("Select Ticker", sorted(df['Ticker'].unique()), key="eda_dist_ticker")
        t_data = df[df['Ticker'] == ticker_sel].copy()
        t_data['daily_return'] = t_data['Close'].pct_change() * 100
        t_data = t_data.dropna(subset=['daily_return'])

        fig = px.histogram(t_data, x='daily_return', nbins=60, template=tmpl,
                           color_discrete_sequence=[accent], labels={'daily_return': 'Daily Return (%)'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Return", f"{t_data['daily_return'].mean():.3f}%")
        c2.metric("Std Dev", f"{t_data['daily_return'].std():.3f}%")
        c3.metric("Max Return", f"{t_data['daily_return'].max():.2f}%")
        c4.metric("Min Return", f"{t_data['daily_return'].min():.2f}%")

        st.markdown("#### Rolling Volatility (20-day)")
        t_data['volatility'] = t_data['daily_return'].rolling(20).std()
        if 'Date' in t_data.columns:
            fig2 = px.line(t_data, x='Date', y='volatility', template=tmpl, color_discrete_sequence=[red])
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig2, width="stretch")


def _render_correlation(df, tmpl):
    st.markdown("#### Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_use = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in numeric_cols]
    if not cols_to_use:
        cols_to_use = numeric_cols[:8]

    corr = df[cols_to_use].corr()
    fig = px.imshow(corr, text_auto='.2f', template=tmpl, color_continuous_scale='RdBu_r', aspect='auto')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450)
    st.plotly_chart(fig, width="stretch")


def _render_insights():
    st.markdown("#### 💡 Key Insights from EDA")
    insights = [
        "📈 **Stock prices** show varying trends across tickers, reflecting different market positions.",
        "📊 **Daily returns** follow an approximately normal distribution with heavier tails.",
        "🔗 **OHLC prices** are highly correlated (>0.99), as expected for intraday price data.",
        "📉 **Volume** shows periodic spikes, often corresponding to earnings and major events.",
        "⚡ **Volatility** fluctuates over time, with increases during uncertainty periods.",
        "🔄 **Moving averages** (7, 30, 100 day) provide useful trend indicators.",
    ]
    for i in insights:
        st.markdown(i)


def _load_data():
    cache_key = "_eda_dataset"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_stock_dataset_2026.csv')
    df = None
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        try:
            from src.backend.data_fetcher import fetch_live_stock_data
            frames = []
            for t in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'JPM', 'JNJ', 'XOM']:
                d = fetch_live_stock_data(t, '2y')
                if d is not None and not d.empty:
                    d = d.reset_index()
                    d['Ticker'] = t
                    frames.append(d)
            if frames:
                df = pd.concat(frames, ignore_index=True)
        except Exception:
            pass

    if df is not None and not df.empty:
        st.session_state[cache_key] = df
        return df
    return pd.DataFrame()

