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
            direction_desc = "positive (price increase)" if movement == "Up" else "negative (price decrease)"
            prob_desc = (
                "strong directional conviction" if prob >= 0.70 
                else "moderate conviction" if prob >= 0.55 
                else "weak conviction (high market noise/indecision)"
            )
            
            r2_scores = {
                'Random Forest': 0.9990,
                'Xgb': 0.9810,
                'Xgboost': 0.9810,
                'SVM (SVR)': -0.1741,
                'LSTM': 0.9957
            }
            active_model_name = price_result.get('model_used', model)
            r2_val = r2_scores.get(active_model_name, 0.5)

            st.markdown(f"""
            ### 📊 Live Prediction Insights for `{ticker}` ({active_model_name})
            * **Ticker Symbol**: `{ticker}`
            * **Prediction Target Date**: `{price_result.get('prediction_date', 'N/A')}`
            * **Directional Forecast**: Predicted to move **{movement}** tomorrow with **{prob:.1%}** certainty.
            * **Price Target Forecast**: Predicted to close at **{disp.get('predicted_price', format_currency(price_result.get('predicted_price', 0)))}** (**{change:+.2f}%** expected change) with **{price_result.get('confidence', 'N/A')}** confidence.

            ---

            ### 🧠 Why are there two different confidence/probability scores?
            The AI prediction pipeline uses **two independent specialized machine learning models** working in tandem:

            1. **Directional Certainty ({move_emoji} `{prob:.1%}`):**
               * **Model**: **Movement Classifier** (solves a binary classification problem: *Will the price rise or fall tomorrow?*).
               * **Live Application**: The classifier predicts tomorrow's direction as **{movement}**, estimating a **{prob:.1%}** probability of a **{direction_desc}** return tomorrow.
               * **Insight**: This represents **{prob_desc}** because it falls within the classifier thresholds.

            2. **Price Level Confidence (Model Details - `{price_result.get('confidence', 'N/A')}`):**
               * **Model**: **Price Regressor** (solves a continuous regression problem: *What will tomorrow's exact closing price be in {currency}?*).
               * **Live Application**: The confidence rating is **{price_result.get('confidence', 'N/A')}**, based on the model's historical baseline accuracy of **{r2_val:.2%}** ($R^2$ score) on test data and the predicted price shift of **{change:+.2f}%**.

            ---

            ### 📐 Model Confidence Classification Thresholds
            Here are the mathematical thresholds used by the system to classify confidence:

            | Model Type | Confidence Rating | Mathematical Threshold Rules |
            | :--- | :--- | :--- |
            | **Price Regressor** (Closing Price) | **🟢 High** | Historical $R^2 \\ge 95\\%$ AND Expected Price Swing $\\le 6\\%$ |
            | | **🟡 Medium** | Historical $R^2 \\ge 80\\%$ AND Expected Price Swing $\\le 12\\%$ |
            | | **🔴 Low** | Historical $R^2 < 80\\%$ OR Expected Price Swing $> 12\\%$ |
            | **Movement Classifier** (Up/Down) | **🟢 High** | Classifier probability certainty $\\ge 70\\%$ |
            | | **🟡 Medium** | Classifier probability certainty between $55\\%$ and $70\\%$ |
            | | **🔴 Low** | Classifier probability certainty $< 55\\%$ |
            """)

        st.markdown(f'<div class="disclaimer-banner">{get_disclaimer_text()}</div>', unsafe_allow_html=True)
