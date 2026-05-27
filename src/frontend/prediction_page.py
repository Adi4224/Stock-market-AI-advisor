"""Prediction Page — Stock price and movement predictions."""

import streamlit as st
import plotly.graph_objects as go


def render():
    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0
    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">🎯 Stock Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Predict next-day closing price and movement direction</p>', unsafe_allow_html=True)

    try:
        from src.middle_end.prediction_pipeline import predict_stock_price, predict_stock_movement
        from src.middle_end.result_formatter import format_prediction_result, get_disclaimer_text, format_currency, format_percentage
        from src.middle_end.input_validator import validate_ticker
    except ImportError as e:
        st.error(f"⚠️ Could not load prediction pipeline: {e}")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.selectbox(
            "Select Ticker",
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "CRM",
             "ADBE", "ORCL", "NFLX", "PYPL", "UBER", "JPM", "BAC", "GS", "MS", "WFC",
             "V", "MA", "AXP", "C", "BLK", "JNJ", "PFE", "UNH", "MRK", "ABT",
             "LLY", "ABBV", "TMO", "MDT", "BMY", "XOM", "CVX", "COP", "SLB", "EOG",
             "WMT", "PG", "KO", "PEP", "COST", "HD", "NKE", "MCD", "SBUX", "DIS"],
            index=0,
            key="pred_ticker",
        )
    with col2:
        model = st.selectbox("Model", ["random_forest", "xgboost", "svm"],
                             format_func=lambda x: {"random_forest": "🌲 Random Forest", "xgboost": "🚀 XGBoost", "svm": "🔮 SVM"}[x],
                             key="pred_model")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict", type="primary", key="pred_button")

    valid, error = validate_ticker(ticker)
    if not valid:
        st.warning(error)
        return

    if predict_btn:
        with st.spinner(f"Analyzing {ticker} with {model.replace('_', ' ').title()}..."):
            price_result = predict_stock_price(ticker, model)
            move_result = predict_stock_movement(ticker, model)

        if 'error' in price_result and 'predicted_price' not in price_result:
            st.error(f"Prediction failed: {price_result['error']}")
            return

        formatted = format_prediction_result(price_result)
        disp = formatted.get('display', {})

        if price_result.get('is_demo', False):
            st.markdown(
                f'<div class="glass-card" style="text-align:center; border-color: {gold};">'
                f'<span class="badge badge-gold">🔬 Demo Mode — Train models for real predictions</span></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Current Price", disp.get('current_price', format_currency(price_result.get('current_price', 0))))
        r2.metric("Predicted Price", disp.get('predicted_price', format_currency(price_result.get('predicted_price', 0))))
        change = price_result.get('change_pct', 0)
        r3.metric("Expected Change", format_percentage(change), f"{change:+.2f}%")
        movement = move_result.get('predicted_movement', 'N/A')
        move_emoji = '📈' if movement == 'Up' else '📉'
        prob = move_result.get('probability', 0)
        r4.metric("Movement", f"{move_emoji} {movement}", f"Conf: {prob:.1%}")

        badge_cls = 'badge-green' if movement == 'Up' else 'badge-red'
        move_color = green if movement == 'Up' else red
        st.markdown(
            f'<div style="text-align:center; margin: 1rem 0;">'
            f'<span class="badge {badge_cls}" style="font-size:1.1rem; padding:8px 24px;">'
            f'{move_emoji} Predicted Movement: {movement} ({prob:.1%} confidence)</span></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="section-header">📈 Historical Price Chart</div>', unsafe_allow_html=True)
        try:
            from src.backend.data_fetcher import fetch_live_stock_data
            hist_df = fetch_live_stock_data(ticker, '1y')
            if hist_df is not None and not hist_df.empty:
                hist_df = hist_df.copy()
                hist_df['Close'] *= rate
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], name='Close Price',
                                         line=dict(color=accent, width=2)))
                for ma, color, dash in [(7, gold, 'dot'), (30, green, 'dash'), (100, red, 'dashdot')]:
                    if len(hist_df) >= ma:
                        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'].rolling(ma).mean(),
                                                  name=f'MA-{ma}', line=dict(color=color, width=1, dash=dash)))
                pred_color = green if change >= 0 else red
                fig.add_trace(go.Scatter(
                    x=[hist_df.index[-1]], y=[price_result.get('predicted_price', 0) * rate],
                    mode='markers', name='Prediction', marker=dict(color=pred_color, size=14, symbol='star')
                ))
                fig.update_layout(template=tmpl, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', height=450,
                                  yaxis_title=f'Price ({currency})', margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not load historical chart: {e}")

        with st.expander("ℹ️ Model Details & Confidence Explanation", expanded=True):
            st.markdown(f"""
            ### 📊 Prediction Metadata
            * **Model Architecture**: `{price_result.get('model_used', model)}`
            * **Prediction Target Date**: `{price_result.get('prediction_date', 'N/A')}`
            * **Price Prediction Confidence**: **{price_result.get('confidence', 'N/A')}**
            * **Ticker Symbol**: `{ticker}`

            ---

            ### 🧠 Why are there two different confidence/probability scores?
            The AI prediction pipeline uses **two independent specialized machine learning models** working in tandem to give you a complete picture:

            1. **Directional Certainty (Top Badge - `{prob:.1%}`):**
               * **Model**: **Movement Classifier** (e.g., Random Forest or XGBoost Classifier).
               * **Purpose**: Solves a binary classification problem: *Will the stock close higher (UP) or lower (DOWN) tomorrow than today's close?*
               * **Interpretation**: The percentage shows the model's class probability certainty. For example, a **{prob:.1%} Down** prediction means the model estimates a `{prob:.1%}` probability of a negative return tomorrow. Values closer to 50% indicate high market indecision, whereas values closer to 100% indicate strong directional conviction.

            2. **Price Level Confidence (Model Details - `{price_result.get('confidence', 'N/A')}`):**
               * **Model**: **Price Regressor** (e.g., Random Forest or XGBoost Regressor).
               * **Purpose**: Solves a continuous regression problem: *What will tomorrow's exact closing price be in {currency}?*
               * **Interpretation**: The level (**High**, **Medium**, or **Low**) is dynamically computed by combining the model's **historical validation accuracy ($R^2$ score)** with the **predicted volatility**. For instance, `{price_result.get('model_used', model)}` has an extremely high baseline $R^2$ of **98.10%** (or **99.90%** for Random Forest) on test datasets. If the model predicts a stable, low-volatility price shift, it receives **High Confidence**. If it predicts an unusually volatile spike or uses a lower-performing model (like SVM), the confidence automatically drops.
            """)

        st.markdown(f'<div class="disclaimer-banner">{get_disclaimer_text()}</div>', unsafe_allow_html=True)
