"""Model Performance Page — Model comparison and evaluation metrics."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">⚙️ Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Compare ML models across evaluation metrics</p>', unsafe_allow_html=True)

    comparison_path = os.path.join(PROJECT_ROOT, 'reports', 'model_comparison.csv')
    has_results = os.path.exists(comparison_path)
    if has_results:
        results_df = pd.read_csv(comparison_path)
        if results_df.dropna(how='all', subset=[c for c in results_df.columns if c not in ['model_name', 'task']]).empty:
            has_results = False

    if not has_results:
        st.info("📊 No trained model results found. Showing sample performance data.")
        st.code("py src/backend/train_random_forest.py\npy src/backend/train_xgboost.py\npy src/backend/train_lstm.py", language="bash")
        results_df = pd.DataFrame([
            {'model_name': 'Random Forest Regressor', 'task': 'Regression', 'mae': 2.34, 'mse': 8.92, 'rmse': 2.99, 'r2': 0.9847},
            {'model_name': 'XGBoost Regressor', 'task': 'Regression', 'mae': 1.89, 'mse': 6.15, 'rmse': 2.48, 'r2': 0.9901},
            {'model_name': 'LSTM', 'task': 'Regression', 'mae': 2.12, 'mse': 7.45, 'rmse': 2.73, 'r2': 0.9878},
            {'model_name': 'Random Forest Classifier', 'task': 'Classification', 'accuracy': 0.6823, 'precision': 0.6912, 'recall': 0.6734, 'f1': 0.6822},
            {'model_name': 'XGBoost Classifier', 'task': 'Classification', 'accuracy': 0.7045, 'precision': 0.7123, 'recall': 0.6956, 'f1': 0.7038},
        ])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Regression", "📋 Classification", "🏆 Best Model"])
    with tab1:
        _render_overview(results_df)
    with tab2:
        _render_regression(results_df, tmpl, accent, red, green)
    with tab3:
        _render_classification(results_df, tmpl, accent, gold, red, green)
    with tab4:
        _render_best_model(results_df, green, gold)


def _render_overview(df):
    st.markdown('<div class="section-header">📊 Model Comparison Table</div>', unsafe_allow_html=True)
    st.dataframe(df.style.format(precision=4, na_rep='-'), width="stretch", hide_index=True)
    reg = df[df['task'] == 'Regression']
    cls = df[df['task'] == 'Classification']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regression Models", str(len(reg)))
    c2.metric("Classification Models", str(len(cls)))
    c3.metric("Best R²", f"{reg['r2'].max():.4f}" if 'r2' in reg.columns and not reg.empty else "N/A")
    c4.metric("Best F1", f"{cls['f1'].max():.4f}" if 'f1' in cls.columns and not cls.empty else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📚 Evaluation Metrics Guide</div>', unsafe_allow_html=True)
    
    col_reg, col_cls = st.columns(2)
    with col_reg:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
        <h4 style="margin-top:0; color:#818CF8;">📈 Regression Metrics (Price Level Forecasting)</h4>
        <ul style="padding-left: 20px; line-height: 1.6;">
            <li><b>Mean Absolute Error (MAE)</b>:<br>
                <span class="muted">The average absolute difference between predicted and actual stock prices. Represents the expected average currency error per trade. Lower is better.</span>
            </li><br>
            <li><b>Mean Squared Error (MSE)</b>:<br>
                <span class="muted">The average of squared errors. By squaring the differences, it heavily penalizes larger, highly volatile prediction errors. Lower is better.</span>
            </li><br>
            <li><b>Root Mean Squared Error (RMSE)</b>:<br>
                <span class="muted">The square root of the MSE, bringing the metric back into the stock's actual currency denomination. Reflects standard deviation of price residuals. Lower is better.</span>
            </li><br>
            <li><b>Coefficient of Determination ($R^2$ Score)</b>:<br>
                <span class="muted">The proportion of variance in stock prices explained by our technical indicators. 1.00 represents a perfect model, while 0.99 means the model successfully explains 99% of pricing patterns.</span>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_cls:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
        <h4 style="margin-top:0; color:#FBBF24;">📋 Classification Metrics (Directional Forecasting)</h4>
        <ul style="padding-left: 20px; line-height: 1.6;">
            <li><b>Accuracy (Hit Rate)</b>:<br>
                <span class="muted">The percentage of correct directional calls (UP predicted correctly + DOWN predicted correctly) out of total forecasts. Measures basic hit rate. Higher is better.</span>
            </li><br>
            <li><b>Precision (Trade Selection Certainty)</b>:<br>
                <span class="muted">The ratio of correct positive forecasts (correct UP calls) to all positive forecasts made. High precision reduces the risk of making false-alarm buying trades.</span>
            </li><br>
            <li><b>Recall (Sensitivity to Opportunities)</b>:<br>
                <span class="muted">The ratio of correct positive forecasts to all actual upward market trading sessions. High recall ensures the model does not miss profitable market surges.</span>
            </li><br>
            <li><b>F1-Score (Robustness Index)</b>:<br>
                <span class="muted">The harmonic mean of Precision and Recall. The single most balanced classification metric, critical for quantitative models when false buy-signals and missed opportunities are both costly.</span>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def _render_regression(df, tmpl, accent, red, green):
    reg = df[df['task'] == 'Regression'].copy()
    if reg.empty:
        st.info("No regression results.")
        return
    st.markdown('<div class="section-header">📈 Regression Metrics</div>', unsafe_allow_html=True)
    metrics = ['mae', 'rmse', 'r2']
    colors = [accent, red, green]
    cols = st.columns(3)
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in reg.columns:
            with cols[i]:
                fig = go.Figure(go.Bar(x=reg['model_name'], y=reg[metric], marker_color=color,
                                       text=reg[metric].round(4), textposition='outside'))
                fig.update_layout(title=metric.upper(), template=tmpl, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=10, r=10, t=40, b=80),
                                  xaxis_tickangle=-30)
                st.plotly_chart(fig, width="stretch")


def _render_classification(df, tmpl, accent, gold, red, green):
    cls = df[df['task'] == 'Classification'].copy()
    if cls.empty:
        st.info("No classification results.")
        return
    st.markdown('<div class="section-header">📋 Classification Metrics</div>', unsafe_allow_html=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = [accent, gold, red, green]
    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        if metric in cls.columns:
            fig.add_trace(go.Bar(name=metric.capitalize(), x=cls['model_name'], y=cls[metric],
                                 marker_color=color, text=cls[metric].round(4), textposition='outside'))
    fig.update_layout(barmode='group', template=tmpl, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', height=450, yaxis_range=[0, 1.1],
                      margin=dict(l=10, r=10, t=20, b=80))
    st.plotly_chart(fig, width="stretch")


def _render_best_model(df, green, gold):
    st.markdown('<div class="section-header">🏆 Best Model Selection</div>', unsafe_allow_html=True)
    reg = df[df['task'] == 'Regression']
    cls = df[df['task'] == 'Classification']
    if not reg.empty and 'rmse' in reg.columns:
        best = reg.loc[reg['rmse'].idxmin()]
        st.markdown(
            f'<div class="glass-card" style="border-left: 4px solid {green};">'
            f'<h4 style="color:{green};">🎯 Best Regression: {best["model_name"]}</h4>'
            f'<p>Lowest RMSE ({best["rmse"]:.4f}) and highest R² ({best.get("r2", "N/A")})</p></div>',
            unsafe_allow_html=True)
    if not cls.empty and 'f1' in cls.columns:
        best = cls.loc[cls['f1'].idxmax()]
        st.markdown(
            f'<div class="glass-card" style="border-left: 4px solid {gold};">'
            f'<h4 style="color:{gold};">🎯 Best Classification: {best["model_name"]}</h4>'
            f'<p>Highest F1 Score ({best["f1"]:.4f})</p></div>',
            unsafe_allow_html=True)
    st.markdown("""
    #### 📝 Model Selection Rationale
    - **Regression**: RMSE penalizes larger errors more heavily — critical for price prediction
    - **Classification**: F1 balances precision and recall — important when both false positives/negatives matter
    - **LSTM**: Captures temporal patterns that tree-based models may miss
    - All models use **TimeSeriesSplit** to prevent look-ahead bias
    """)
