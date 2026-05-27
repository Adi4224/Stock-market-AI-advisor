"""Customer Analytics Page — User profiling and risk assessment."""

import streamlit as st
import plotly.graph_objects as go


def render():
    currency = st.session_state.get("currency", "USD")
    rate = 83.5 if currency == "INR" else 1.0
    symbol = "₹" if currency == "INR" else "$"

    C = st.session_state.get("_colors", {})
    tmpl = st.session_state.get("_chart_template", "plotly_dark")
    accent = C.get("accent", "#818CF8")
    green = C.get("green", "#34D399")
    red = C.get("red", "#F87171")
    gold = C.get("gold", "#FBBF24")

    st.markdown('<h1 class="gradient-text" style="font-size:2rem;">👤 Customer Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="muted">Build your investor profile and discover your risk category</p>', unsafe_allow_html=True)

    try:
        from src.middle_end.customer_profile_pipeline import create_user_profile, get_investor_description, get_risk_breakdown
        from src.middle_end.result_formatter import get_disclaimer_text
    except ImportError as e:
        st.error(f"⚠️ Could not load customer analytics pipeline: {e}")
        return

    with st.form("user_profile_form"):
        st.markdown('<div class="section-header">📋 Your Investment Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
            income = st.number_input(f"Annual Income ({symbol})", min_value=0, max_value=int(10000000 * rate), value=int(75000 * rate), step=int(5000 * rate))
            risk_tolerance = st.slider("Risk Tolerance", min_value=1, max_value=10, value=5,
                                       help="1 = Very Conservative, 10 = Very Aggressive")
            investment_horizon = st.selectbox("Investment Horizon", ["short", "medium", "long"],
                                              format_func=lambda x: {"short": "Short-term (< 1 year)", "medium": "Medium-term (1-5 years)",
                                                                     "long": "Long-term (5+ years)"}[x])
        with c2:
            preferred_sectors = st.multiselect("Preferred Sectors",
                                               ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Entertainment"],
                                               default=["Technology"])
            portfolio_size = st.number_input(f"Portfolio Size ({symbol})", min_value=0, max_value=int(100000000 * rate), value=int(25000 * rate), step=int(1000 * rate))
            trading_frequency = st.selectbox("Trading Frequency", ["daily", "weekly", "monthly", "quarterly"],
                                              format_func=str.capitalize, index=2)
            past_returns = st.slider("Historical Returns (%)", min_value=-30, max_value=50, value=8)

        submitted = st.form_submit_button("🔍 Analyze My Profile", type="primary")

    if submitted:
        user_inputs = {
            'age': age, 'income': income / rate, 'risk_tolerance': risk_tolerance,
            'investment_horizon': investment_horizon, 'preferred_sectors': ', '.join(preferred_sectors),
            'portfolio_size': portfolio_size / rate, 'trading_frequency': trading_frequency,
            'past_returns': past_returns
        }

        with st.spinner("Analyzing your profile..."):
            profile = create_user_profile(user_inputs)
            breakdown = get_risk_breakdown(user_inputs)

        st.session_state['user_profile'] = profile

        investor_type = profile.get('investor_type', 'Moderate')
        risk_score = profile.get('risk_score', 5)
        type_colors = {'Conservative': accent, 'Moderate': gold, 'Aggressive': red}
        type_icons = {'Conservative': '🛡️', 'Moderate': '⚖️', 'Aggressive': '🚀'}
        color = type_colors.get(investor_type, gold)
        icon = type_icons.get(investor_type, '⚖️')

        st.markdown(
            f'<div class="glass-card animate-fade-in" style="text-align:center; border-top: 4px solid {color};">'
            f'<div style="font-size:3.5rem;">{icon}</div>'
            f'<h2 style="color:{color}; margin:8px 0;">{investor_type} Investor</h2>'
            f'<p class="muted">Risk Score: {risk_score}/10 — {profile.get("risk_level", "Medium")}</p></div>',
            unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### 📊 Risk Score Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=risk_score, title={'text': 'Risk Score'},
                delta={'reference': 5},
                gauge={'axis': {'range': [0, 10]}, 'bar': {'color': color},
                       'steps': [{'range': [0, 3.5], 'color': f'rgba({_hex_to_rgb(accent)},0.12)'},
                                 {'range': [3.5, 6.5], 'color': f'rgba({_hex_to_rgb(gold)},0.12)'},
                                 {'range': [6.5, 10], 'color': f'rgba({_hex_to_rgb(red)},0.12)'}]}
            ))
            fig.update_layout(template=tmpl, paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        with col_r:
            st.markdown("#### 🕸️ Profile Attributes")
            categories = ['Risk Tolerance', 'Trading Activity', 'Returns History', 'Portfolio Size', 'Income Level']
            values = [
                risk_tolerance / 10,
                {'daily': 1.0, 'weekly': 0.75, 'monthly': 0.5, 'quarterly': 0.25}[trading_frequency],
                max(0, (past_returns + 30) / 80),
                min(portfolio_size / 500000, 1.0),
                min(income / 300000, 1.0),
            ]
            values.append(values[0])
            categories.append(categories[0])

            fig2 = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself',
                                              fillcolor=f'rgba({_hex_to_rgb(color)},0.15)',
                                              line=dict(color=color, width=2)))
            fig2.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0, 1])),
                               template=tmpl, paper_bgcolor='rgba(0,0,0,0)', height=300,
                               margin=dict(l=40, r=40, t=40, b=40), showlegend=False)
            st.plotly_chart(fig2, width="stretch")

        st.markdown('<div class="section-header">📝 Profile Summary</div>', unsafe_allow_html=True)
        st.info(profile.get('profile_summary', ''))

        with st.expander("📊 Risk Score Breakdown"):
            for comp, data in breakdown.items():
                if isinstance(data, dict):
                    st.markdown(f"**{comp.replace('_', ' ').title()}**: Raw={data['value']:.1f} × Weight={data['weight']} = **{data['weighted']:.2f}**")

        st.markdown('<div class="section-header">🎯 Recommended Strategy</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="glass-card" style="border-left: 4px solid {color};">'
            f'{profile.get("recommended_strategy", "Diversify your portfolio.")}</div>',
            unsafe_allow_html=True)

        rec_sectors = profile.get('recommended_sectors', [])
        if rec_sectors:
            st.markdown("#### 🏢 Recommended Sectors")
            sector_cols = st.columns(len(rec_sectors))
            for col, sector in zip(sector_cols, rec_sectors):
                col.markdown(f'<div class="glass-card" style="text-align:center; padding:12px;"><strong>{sector}</strong></div>',
                             unsafe_allow_html=True)

        st.markdown(f'<div class="disclaimer-banner">{get_disclaimer_text()}</div>', unsafe_allow_html=True)


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)}"
