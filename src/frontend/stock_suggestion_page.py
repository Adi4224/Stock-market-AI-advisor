"""Stock Suggestion Page — Personalized stock recommendations."""

import streamlit as st
import pandas as pd


def render():
    C = st.session_state.get("_colors", {})
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">💡 Stock Suggestions</h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Personalized stock recommendations based on your investment profile</p>', unsafe_allow_html=True)

    try:
        from src.middle_end.stock_suggestion_engine import suggest_stocks, DISCLAIMER
        from src.middle_end.result_formatter import format_suggestion_result, get_disclaimer_text
        from src.middle_end.customer_profile_pipeline import create_user_profile
    except ImportError as e:
        st.error(f"⚠️ Could not load suggestion engine: {e}")
        return

    st.markdown(
        '<div class="disclaimer-banner" style="text-align:center; font-size:0.85rem; margin-bottom:1.5rem;">'
        '⚠️ <strong>IMPORTANT:</strong> These suggestions are for <em>educational purposes only</em> '
        'and do NOT constitute financial advice.</div>',
        unsafe_allow_html=True)

    profile = st.session_state.get('user_profile', None)

    if profile is None:
        st.info("📋 No investor profile found. Create one below or visit the **Customer Analytics** page.")
        with st.expander("⚡ Quick Profile Setup", expanded=True):
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                risk = st.slider("Risk Tolerance", 1, 10, 5, key="quick_risk")
            with qc2:
                horizon = st.selectbox("Horizon", ["short", "medium", "long"],
                                        format_func=str.capitalize, index=1, key="quick_horizon")
            with qc3:
                freq = st.selectbox("Frequency", ["daily", "weekly", "monthly", "quarterly"],
                                     format_func=str.capitalize, index=2, key="quick_freq")
            if st.button("🚀 Get Suggestions", type="primary", key="quick_suggest"):
                profile = create_user_profile({
                    'age': 35, 'income': 75000, 'risk_tolerance': risk,
                    'investment_horizon': horizon, 'preferred_sectors': 'Technology, Finance',
                    'portfolio_size': 25000, 'trading_frequency': freq, 'past_returns': 8,
                })
                st.session_state['user_profile'] = profile

    if profile is None:
        return

    investor_type = profile.get('investor_type', 'Moderate')
    type_colors = {'Conservative': accent, 'Moderate': gold, 'Aggressive': red}
    type_icons = {'Conservative': '🛡️', 'Moderate': '⚖️', 'Aggressive': '🚀'}
    color = type_colors.get(investor_type, gold)

    st.markdown(
        f'<div class="glass-card" style="border-left: 4px solid {color}; margin-bottom: 1rem;">'
        f'<strong>{type_icons.get(investor_type, "⚖️")} Your Profile: {investor_type} Investor</strong>'
        f'<span class="muted" style="margin-left:12px;">Risk Score: {profile.get("risk_score", 5)}/10</span></div>',
        unsafe_allow_html=True)

    with st.spinner("Finding the best stocks for your profile..."):
        suggestions = suggest_stocks(profile)
        formatted = format_suggestion_result(suggestions)

    st.markdown(f'<div class="section-header">📊 Recommended Stocks ({len(formatted)} matches)</div>', unsafe_allow_html=True)

    risk_color_map = {'Conservative': accent, 'Moderate': gold, 'Aggressive': red}

    for i in range(0, len(formatted), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(formatted):
                s = formatted[i + j]
                risk_level = s.get('risk_level', 'Moderate')
                r_color = risk_color_map.get(risk_level, gold)
                with col:
                    st.markdown(
                        f'<div class="glass-card" style="border-top: 3px solid {r_color}; min-height:200px;">'
                        f'<h4 style="color:{r_color}; margin:0 0 4px;">{s.get("ticker", "")}</h4>'
                        f'<p style="margin:0 0 8px; font-size:0.9rem;">{s.get("company", "")}</p>'
                        f'<div style="margin-bottom:8px;">'
                        f'<span class="badge badge-accent">{s.get("sector", "")}</span> '
                        f'<span class="badge" style="background:rgba({_hex_to_rgb(r_color)},0.12); color:{r_color};">{risk_level}</span>'
                        f'</div>'
                        f'<p class="muted" style="font-size:0.8rem;">{s.get("reason", "")}</p>'
                        f'<p class="muted" style="font-size:0.72rem;">Volatility: {s.get("expected_volatility", "N/A")}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Comparison Table</div>', unsafe_allow_html=True)
    table_data = [{'Ticker': s['ticker'], 'Company': s['company'], 'Sector': s['sector'],
                   'Risk': s['risk_level'], 'Volatility': s['expected_volatility']} for s in suggestions]
    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

    st.markdown('<div class="section-header">🎨 Risk Level Legend</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f'<div class="glass-card" style="text-align:center; border-top:3px solid {accent};">🛡️ <strong>Conservative</strong><br><span class="muted">Low risk, stable returns</span></div>', unsafe_allow_html=True)
    lc2.markdown(f'<div class="glass-card" style="text-align:center; border-top:3px solid {gold};">⚖️ <strong>Moderate</strong><br><span class="muted">Balanced risk/return</span></div>', unsafe_allow_html=True)
    lc3.markdown(f'<div class="glass-card" style="text-align:center; border-top:3px solid {red};">🚀 <strong>Aggressive</strong><br><span class="muted">High risk, high potential</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-banner" style="margin-top:1.5rem;">{get_disclaimer_text()}</div>', unsafe_allow_html=True)


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
