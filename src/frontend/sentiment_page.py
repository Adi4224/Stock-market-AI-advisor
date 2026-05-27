"""Sentiment Analysis Page — NLP-based financial news sentiment analysis."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def render():
    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">💬 Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Analyze financial news sentiment using VADER NLP</p>', unsafe_allow_html=True)

    try:
        from src.middle_end.sentiment_pipeline import analyze_news_sentiment, get_ticker_sentiment, get_market_sentiment
        from src.middle_end.result_formatter import format_sentiment_result
    except ImportError as e:
        st.error(f"⚠️ Could not load sentiment pipeline: {e}")
        return

    tab1, tab2, tab3 = st.tabs(["📝 Analyze Text", "📊 Ticker Sentiment", "🌐 Market Overview"])

    with tab1:
        _render_text_analysis(analyze_news_sentiment, format_sentiment_result, tmpl, accent, green, red, gold)
    with tab2:
        _render_ticker_sentiment(get_ticker_sentiment, tmpl, green, red, gold)
    with tab3:
        _render_market_overview(get_market_sentiment, tmpl)


def _render_text_analysis(analyzer, formatter, tmpl, accent, green, red, gold):
    st.markdown('<div class="section-header">📝 Enter a Financial News Headline</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="glass-card" style="margin-bottom: 1.5rem;">'
        '<h4>💡 How it works: Custom Text Sentiment</h4>'
        '<p class="muted" style="margin: 0; font-size: 0.88rem;">'
        'This utility uses natural language processing (NLP) to evaluate news text in real-time. '
        'It tokenizes the input text and applies VADER (Valence Aware Dictionary and Sentiment Reasoner) '
        'lexicon-based scoring to calculate positive, neutral, and negative sentiment ratios. '
        'A final compound score ranging from -1.0 (highly bearish) to +1.0 (highly bullish) is generated to assist '
        'in quantifying retail news sentiment impact.'
        '</p></div>',
        unsafe_allow_html=True
    )

    samples = [
        "Apple reports record Q4 earnings, beating Wall Street estimates by 15%",
        "Tesla shares plunge 8% after disappointing delivery numbers",
        "Federal Reserve announces interest rates remain unchanged for Q2",
        "NVIDIA surpasses expectations with AI chip revenue growth of 200%",
        "Bank of America faces regulatory scrutiny over lending practices",
    ]

    if "sentiment_text_input" not in st.session_state:
        st.session_state["sentiment_text_input"] = ""

    st.markdown("**Quick Samples:**")
    sample_cols = st.columns(len(samples))
    for i, (col, s) in enumerate(zip(sample_cols, samples)):
        with col:
            if st.button(f"📰 #{i+1}", key=f"sample_{i}"):
                st.session_state["sentiment_text_input"] = s
                st.rerun()

    text_input = st.text_area("News Headline / Text", height=100,
                              value=st.session_state["sentiment_text_input"],
                              placeholder="Enter a financial news headline to analyze...")
    st.session_state["sentiment_text_input"] = text_input

    if st.button("🔍 Analyze Sentiment", type="primary", key="analyze_btn") and text_input.strip():
        with st.spinner("Analyzing..."):
            result = analyzer(text_input)
            formatted = formatter(result)
            disp = formatted.get('display', {})

        label = result.get('sentiment_label', 'Neutral')
        score = result.get('sentiment_score', 0)
        emoji_map = {'Positive': '😊', 'Negative': '😟', 'Neutral': '😐'}
        color_map = {'Positive': green, 'Negative': red, 'Neutral': gold}
        emoji = emoji_map.get(label, '😐')
        color = color_map.get(label, gold)

        st.markdown(
            f'<div class="glass-card animate-fade-in" style="text-align:center; border-top: 4px solid {color};">'
            f'<div style="font-size:4rem; margin-bottom:10px;">{emoji}</div>'
            f'<h2 style="color:{color}; margin:0;">{label}</h2>'
            f'<p class="muted" style="font-size:1.1rem;">Score: {score:+.4f}</p></div>',
            unsafe_allow_html=True)

        gauge_value = (score + 1) / 2 * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=gauge_value,
            title={'text': 'Sentiment Gauge'},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                   'steps': [{'range': [0, 35], 'color': f'rgba({_hex_to_rgb(red)},0.15)'},
                             {'range': [35, 65], 'color': f'rgba({_hex_to_rgb(gold)},0.15)'},
                             {'range': [65, 100], 'color': f'rgba({_hex_to_rgb(green)},0.15)'}],
                   'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.8, 'value': gauge_value}}
        ))
        fig.update_layout(template=tmpl, paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        details = result.get('details', {})
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Positive", f"{details.get('positive', 0):.1%}")
        bc2.metric("Neutral", f"{details.get('neutral', 0):.1%}")
        bc3.metric("Negative", f"{details.get('negative', 0):.1%}")


def _render_ticker_sentiment(get_sentiment, tmpl, green, red, gold):
    st.markdown(
        '<div class="glass-card" style="margin-bottom: 1.5rem;">'
        '<h4>💡 How it works: Ticker-Specific News Aggregation</h4>'
        '<p class="muted" style="margin: 0; font-size: 0.88rem;">'
        'This module queries a database of aggregated financial news articles linked to the selected company ticker. '
        'It calculates the average sentiment compound score across all matched articles and charts the relative '
        'distribution of Positive, Neutral, and Negative reports. It also displays the individual headlines with '
        'their calculated scores to provide granular sentiment visibility for individual stock assets.'
        '</p></div>',
        unsafe_allow_html=True
    )

    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "JNJ"]
    ticker = st.selectbox("Select Ticker", tickers, key="sent_ticker")

    if st.button("📊 Get Sentiment", type="primary", key="ticker_sent_btn"):
        with st.spinner(f"Analyzing sentiment for {ticker}..."):
            result = get_sentiment(ticker)

        overall = result.get('overall_sentiment', 'Neutral')
        color_map = {'Positive': green, 'Negative': red, 'Neutral': gold}
        emoji_map = {'Positive': '😊', 'Negative': '😟', 'Neutral': '😐'}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall", f"{emoji_map.get(overall, '😐')} {overall}")
        c2.metric("Avg Score", f"{result.get('avg_score', 0):+.4f}")
        c3.metric("Total Articles", str(result.get('total_articles', 0)))
        c4.metric("Pos/Neg", f"{result.get('positive_count', 0)}/{result.get('negative_count', 0)}")

        labels = ['Positive', 'Neutral', 'Negative']
        values = [result.get('positive_count', 0), result.get('neutral_count', 0), result.get('negative_count', 0)]
        fig = px.pie(values=values, names=labels, color=labels,
                     color_discrete_map={'Positive': green, 'Neutral': gold, 'Negative': red},
                     template=tmpl, hole=0.4)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig, width="stretch")

        headlines = result.get('recent_headlines', [])
        if headlines:
            st.markdown("#### Recent Headlines")
            for h in headlines[:8]:
                label = h.get('sentiment_label', 'Neutral')
                color = color_map.get(label, gold)
                st.markdown(
                    f'<div class="glass-card" style="padding:12px 16px; margin-bottom:8px; border-left: 3px solid {color};">'
                    f'<strong>{h.get("headline", "")}</strong><br>'
                    f'<span class="muted">{h.get("date", "")} | '
                    f'<span style="color:{color};">{label} ({h.get("sentiment_score", 0):+.3f})</span></span></div>',
                    unsafe_allow_html=True)


def _render_market_overview(get_sentiment, tmpl):
    st.markdown(
        '<div class="glass-card" style="margin-bottom: 1.5rem;">'
        '<h4>💡 How it works: Market-Wide Sentiment Leaderboard</h4>'
        '<p class="muted" style="margin: 0; font-size: 0.88rem;">'
        'This overview scans and summarizes retail market sentiment indicators across the entire universe of tracked tickers. '
        'By averaging compound sentiment values from all recent financial articles globally, it highlights market momentum '
        'and presents a sorted ranked leaderboard, letting you instantly identify which companies are currently receiving '
        'the most favorable or unfavorable news coverage.'
        '</p></div>',
        unsafe_allow_html=True
    )

    if st.button("🌐 Load Market Sentiment", type="primary", key="market_sent_btn"):
        with st.spinner("Analyzing overall market sentiment..."):
            result = get_sentiment()
        overall = result.get('overall_sentiment', 'Neutral')
        st.markdown(f"### Overall Market Sentiment: **{overall}**")
        st.metric("Average Score", f"{result.get('avg_compound_score', result.get('avg_score', 0)):+.4f}")
        breakdown = result.get('ticker_breakdown', {})
        if breakdown:
            st.markdown("#### Sentiment by Ticker")
            rows = [{'Ticker': t, 'Sentiment': d.get('label', 'N/A'),
                     'Avg Score': d.get('avg_score', 0), 'Articles': d.get('count', 0)}
                    for t, d in breakdown.items()]
            st.dataframe(pd.DataFrame(rows).sort_values('Avg Score', ascending=False), width="stretch")


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
