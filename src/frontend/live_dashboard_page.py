"""Live Dashboard Page — Real-time stock data with interactive charts."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.middle_end.result_formatter import format_currency



def _fetch_stock(ticker, period):
    """Fetch stock data — uses session_state for caching to avoid decorator issues."""
    cache_key = f"_stock_cache_{ticker}_{period}"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    df = None
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period)
    except Exception:
        pass

    if df is not None and not df.empty:
        st.session_state[cache_key] = df

    return df


def render():
    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0
    symbol = "₹" if currency == "INR" else "$"

    st.markdown(
        '<h1 class="gradient-text animate-fade-in" style="font-size:2rem; margin-bottom:0;">📊 Live Stock Dashboard</h1>'
        '<p class="muted" style="margin-top:2px;">Real-time market data powered by yfinance</p>',
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        tickers = st.multiselect(
            "Select Tickers",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "CRM",
             "ADBE", "ORCL", "NFLX", "PYPL", "UBER",
             "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP", "C", "BLK",
             "JNJ", "PFE", "UNH", "MRK", "ABT", "LLY", "ABBV", "TMO", "MDT", "BMY",
             "XOM", "CVX", "COP", "SLB", "EOG",
             "WMT", "PG", "KO", "PEP", "COST", "HD", "NKE", "MCD", "SBUX", "DIS"],
            default=["AAPL", "MSFT", "NVDA"],
            key="dash_tickers",
        )
    with c2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                              index=3, key="dash_period")
    with c3:
        chart_type = st.selectbox("Chart", ["Candlestick", "Line", "Area"],
                                  key="dash_chart_type")

    if not tickers:
        st.info("👆 Select at least one ticker from the dropdown above.")
        return

    # ── Fetch data ───────────────────────────────────────────────────────────
    ticker_data = {}
    progress = st.progress(0, text="Fetching stock data...")
    for i, ticker in enumerate(tickers):
        df = _fetch_stock(ticker, period)
        if df is not None and not df.empty:
            needed = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(c in df.columns for c in needed):
                ticker_data[ticker] = df
        progress.progress((i + 1) / len(tickers), text=f"Loaded {ticker}")
    progress.empty()

    if not ticker_data:
        st.error("❌ Could not fetch data. Check your internet connection.")
        return

    # ── Overview metrics ─────────────────────────────────────────────────────
    cols = st.columns(len(ticker_data))
    for i, (ticker, df) in enumerate(ticker_data.items()):
        cur = float(df['Close'].iloc[-1]) * rate
        prev = float(df['Close'].iloc[-2]) * rate if len(df) > 1 else cur
        pct = ((cur - prev) / prev) * 100 if prev else 0
        with cols[i]:
            st.metric(label=ticker, value=format_currency(cur, convert=False), delta=f"{pct:+.2f}%")

    # ── Charts ───────────────────────────────────────────────────────────────
    if len(ticker_data) == 1:
        t = list(ticker_data.keys())[0]
        _chart(t, ticker_data[t], tmpl, chart_type, accent, green, red, gold)
    else:
        tab_names = list(ticker_data.keys()) + ["📈 Compare"]
        tabs = st.tabs(tab_names)
        for i, t in enumerate(ticker_data.keys()):
            with tabs[i]:
                _chart(t, ticker_data[t], tmpl, chart_type, accent, green, red, gold)
        with tabs[-1]:
            _compare(ticker_data, tmpl)


def _chart(ticker, df, tmpl, chart_type, accent, green, red, gold):
    """Single ticker chart with volume."""
    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0

    df = df.copy()
    df['Open'] *= rate
    df['High'] *= rate
    df['Low'] *= rate
    df['Close'] *= rate

    cur = float(df['Close'].iloc[-1])
    first = float(df['Close'].iloc[0])
    ret = ((cur - first) / first) * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current", format_currency(cur, convert=False))
    m2.metric("High", format_currency(df['Close'].max(), convert=False))
    m3.metric("Low", format_currency(df['Close'].min(), convert=False))
    m4.metric("Avg Vol", f"{df['Volume'].mean():,.0f}")
    m5.metric("Return", f"{ret:+.1f}%")

    dates = df.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=dates, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC', increasing_line_color=green, increasing_fillcolor=green,
            decreasing_line_color=red, decreasing_fillcolor=red,
        ), row=1, col=1)
    elif chart_type == "Area":
        fig.add_trace(go.Scatter(
            x=dates, y=df['Close'], name='Close', fill='tozeroy',
            fillcolor='rgba(99,102,241,0.1)', line=dict(color=accent, width=2),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=dates, y=df['Close'], name='Close', line=dict(color=accent, width=2.5),
        ), row=1, col=1)

    for ma, clr, d in [(7, accent, 'dot'), (30, gold, 'dash'), (100, red, 'dashdot')]:
        if len(df) >= ma:
            fig.add_trace(go.Scatter(
                x=dates, y=df['Close'].rolling(ma).mean(),
                name=f'MA{ma}', line=dict(width=1, color=clr, dash=d),
            ), row=1, col=1)

    vol_c = [green if c >= o else red for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=dates, y=df['Volume'], name='Volume',
                         marker_color=vol_c, opacity=0.45), row=2, col=1)

    fig.update_layout(
        template=tmpl, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=520, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
        legend=dict(orientation='h', y=1.02, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor='rgba(148,163,184,0.08)'),
        yaxis2=dict(gridcolor='rgba(148,163,184,0.08)'),
        yaxis_title=f"Price ({currency})",
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander(f"📊 {ticker} Stats"):
        rets = df['Close'].pct_change().dropna() * 100
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean Return", f"{rets.mean():.3f}%")
        s2.metric("Std Dev", f"{rets.std():.3f}%")
        s3.metric("Best Day", f"+{rets.max():.2f}%")
        s4.metric("Worst Day", f"{rets.min():.2f}%")


def _compare(ticker_data, tmpl):
    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0

    st.markdown('<div class="section-header">📈 Normalized Performance</div>', unsafe_allow_html=True)
    palette = ['#818CF8', '#34D399', '#FBBF24', '#F87171', '#A78BFA', '#FB923C', '#38BDF8']

    fig = go.Figure()
    for i, (t, df) in enumerate(ticker_data.items()):
        norm = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=df.index, y=norm, name=t, mode='lines',
                                 line=dict(width=2.5, color=palette[i % len(palette)])))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.3)")
    fig.update_layout(template=tmpl, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      height=420, yaxis_title='Return (%)', margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation='h', y=1.05))
    st.plotly_chart(fig, width="stretch")

    rows = []
    for t, df in ticker_data.items():
        c = float(df['Close'].iloc[-1]) * rate
        f = float(df['Close'].iloc[0]) * rate
        rows.append({'Ticker': t, 'Price': format_currency(c, convert=False), 'Return': f"{((c-f)/f)*100:+.2f}%",
                     'High': format_currency(df['Close'].max() * rate, convert=False), 'Low': format_currency(df['Close'].min() * rate, convert=False)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
