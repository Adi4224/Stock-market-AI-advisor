"""
Stock Market AI Advisor - Main Streamlit Application
Premium dashboard with light/dark mode, top navigation, and modern design.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Stock Market AI Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  THEME STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "currency" not in st.session_state:
    st.session_state.currency = "USD"


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════
def get_css(theme: str) -> str:
    is_dark = theme == "dark"

    if is_dark:
        bg_primary    = "#0B0F19"
        bg_secondary  = "#111827"
        bg_card       = "rgba(17,24,39,0.85)"
        bg_card_alt   = "rgba(30,41,59,0.7)"
        text_primary  = "#F1F5F9"
        text_muted    = "#94A3B8"
        border_color  = "rgba(99,102,241,0.12)"
        glass_bg      = "rgba(17,24,39,0.6)"
        glass_border  = "rgba(99,102,241,0.18)"
        accent        = "#818CF8"
        accent2       = "#6366F1"
        green         = "#34D399"
        red           = "#F87171"
        gold          = "#FBBF24"
        input_bg      = "#1E293B"
        hover_glow    = "rgba(99,102,241,0.15)"
        gradient_bg   = "linear-gradient(135deg, #0B0F19 0%, #1E1B4B 30%, #111827 70%, #0B0F19 100%)"
        scrollbar_bg  = "#0B0F19"
        navbar_bg     = "rgba(11,15,25,0.88)"
        navbar_border = "rgba(99,102,241,0.15)"
    else:
        bg_primary    = "#F8FAFC"
        bg_secondary  = "#FFFFFF"
        bg_card       = "rgba(255,255,255,0.9)"
        bg_card_alt   = "rgba(241,245,249,0.8)"
        text_primary  = "#0F172A"
        text_muted    = "#64748B"
        border_color  = "rgba(99,102,241,0.08)"
        glass_bg      = "rgba(255,255,255,0.75)"
        glass_border  = "rgba(99,102,241,0.12)"
        accent        = "#6366F1"
        accent2       = "#4F46E5"
        green         = "#059669"
        red           = "#DC2626"
        gold          = "#D97706"
        input_bg      = "#F1F5F9"
        hover_glow    = "rgba(99,102,241,0.08)"
        gradient_bg   = "linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 40%, #F8FAFC 100%)"
        scrollbar_bg  = "#F8FAFC"
        navbar_bg     = "rgba(255,255,255,0.88)"
        navbar_border = "rgba(99,102,241,0.10)"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {{
        --bg-primary:    {bg_primary};
        --bg-secondary:  {bg_secondary};
        --bg-card:       {bg_card};
        --bg-card-alt:   {bg_card_alt};
        --text-primary:  {text_primary};
        --text-muted:    {text_muted};
        --border-color:  {border_color};
        --glass-bg:      {glass_bg};
        --glass-border:  {glass_border};
        --accent:        {accent};
        --accent2:       {accent2};
        --green:         {green};
        --red:           {red};
        --gold:          {gold};
        --input-bg:      {input_bg};
        --hover-glow:    {hover_glow};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }}
    .stApp {{
        background: {gradient_bg};
    }}

    /* ── Hide sidebar completely ─────────────────────────── */
    section[data-testid="stSidebar"] {{ display: none !important; }}
    button[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}

    /* ── Metric Cards ────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: transform 0.25s cubic-bezier(.4,0,.2,1), box-shadow 0.25s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 36px {hover_glow};
    }}
    [data-testid="stMetric"] label {{
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.68rem !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--accent) !important;
        font-weight: 700 !important;
        font-size: 1.35rem !important;
    }}

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
        color: #fff !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.6rem;
        transition: all 0.25s cubic-bezier(.4,0,.2,1);
        letter-spacing: 0.03em;
        font-size: 0.85rem;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99,102,241,0.35);
    }}
    .stButton > button:active {{ transform: translateY(0); }}

    /* ── Tabs ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--bg-card-alt);
        border-radius: 14px;
        padding: 5px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.85rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(79,70,229,0.10));
        color: var(--accent) !important;
        font-weight: 600;
    }}

    /* ── DataFrames ───────────────────────────────────────── */
    .stDataFrame {{
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ── Inputs ───────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }}

    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {scrollbar_bg}; }}
    ::-webkit-scrollbar-thumb {{ background: var(--accent); border-radius: 3px; }}

    /* ── Utility ──────────────────────────────────────────── */
    .gradient-text {{
        background: linear-gradient(135deg, {accent} 0%, {green} 60%, {gold} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }}
    .glass-card {{
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: transform 0.25s cubic-bezier(.4,0,.2,1), box-shadow 0.25s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 14px 44px {hover_glow};
    }}
    .badge {{
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .badge-accent {{ background: rgba(99,102,241,0.15); color: {accent}; }}
    .badge-green  {{ background: rgba(52,211,153,0.15); color: {green}; }}
    .badge-red    {{ background: rgba(248,113,113,0.15); color: {red}; }}
    .badge-gold   {{ background: rgba(251,191,36,0.15); color: {gold}; }}
    .muted {{ color: var(--text-muted); font-size: 0.85rem; }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .animate-fade-in {{ animation: fadeInUp 0.45s ease-out; }}
    .shimmer {{
        background: linear-gradient(135deg, {accent}, {green}, {gold}, {accent});
        background-size: 300% 300%;
        animation: shimmer 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .disclaimer-banner {{
        background: rgba(248,113,113,0.08);
        border: 1px solid rgba(248,113,113,0.2);
        border-radius: 12px;
        padding: 14px 20px;
        font-size: 0.78rem;
        color: {red};
        line-height: 1.5;
    }}
    .section-header {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.5rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-header::after {{
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-color);
        margin-left: 12px;
    }}

    /* ── Hide Streamlit default chrome ────────────────────── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """


# ── Inject CSS ───────────────────────────────────────────────────────────────
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Home",
    "📊 Live Dashboard",
    "🔍 EDA",
    "🎯 Stock Prediction",
    "💬 Sentiment Analysis",
    "👤 Customer Analytics",
    "💡 Stock Suggestions",
    "⚙️ Model Performance",
    "ℹ️ About & Disclaimer",
]

# Top bar: logo + dropdown + theme toggle
bar_left, bar_center, bar_currency, bar_right = st.columns([2, 4.5, 1.5, 0.55])

with bar_left:
    st.markdown(
        '<span class="shimmer" style="font-size:1.1rem; font-weight:800;">📈 Stock AI Advisor</span>',
        unsafe_allow_html=True,
    )

with bar_center:
    page = st.selectbox(
        "Navigate",
        PAGES,
        index=0,
        key="nav_page",
        label_visibility="collapsed",
    )

with bar_currency:
    currency = st.selectbox(
        "Currency",
        ["USD ($)", "INR (₹)"],
        index=0 if st.session_state.get("currency", "USD") == "USD" else 1,
        label_visibility="collapsed",
        key="currency_selectbox_key",
    )
    st.session_state.currency = "USD" if "USD" in currency else "INR"

with bar_right:
    theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")

# Thin separator
st.markdown(
    '<hr style="margin:0.2rem 0 1rem; border:none; border-top:1px solid var(--border-color);">',
    unsafe_allow_html=True,
)

# Disclaimer footer (subtle)
st.markdown(
    '<div style="text-align:center; margin-bottom:0.5rem;">'
    '<span class="muted" style="font-size:0.65rem;">⚠️ Educational purposes only — Not financial advice</span>'
    '</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  THEME HELPERS FOR PAGES
# ═══════════════════════════════════════════════════════════════════════════════
def get_chart_template():
    return "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

def get_theme_colors():
    is_dark = st.session_state.theme == "dark"
    return {
        "accent":  "#818CF8" if is_dark else "#6366F1",
        "accent2": "#6366F1" if is_dark else "#4F46E5",
        "green":   "#34D399" if is_dark else "#059669",
        "red":     "#F87171" if is_dark else "#DC2626",
        "gold":    "#FBBF24" if is_dark else "#D97706",
        "text":    "#F1F5F9" if is_dark else "#0F172A",
        "muted":   "#94A3B8" if is_dark else "#64748B",
        "bg":      "rgba(0,0,0,0)",
        "is_dark": is_dark,
    }

st.session_state._chart_template = get_chart_template()
st.session_state._colors = get_theme_colors()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
PAGE_MAP = {
    "🏠 Home":                "src.frontend.home_page",
    "📊 Live Dashboard":      "src.frontend.live_dashboard_page",
    "🔍 EDA":                 "src.frontend.eda_page",
    "🎯 Stock Prediction":    "src.frontend.prediction_page",
    "💬 Sentiment Analysis":  "src.frontend.sentiment_page",
    "👤 Customer Analytics":  "src.frontend.customer_analytics_page",
    "💡 Stock Suggestions":   "src.frontend.stock_suggestion_page",
    "⚙️ Model Performance":  "src.frontend.performance_page",
    "ℹ️ About & Disclaimer": "src.frontend.about_page",
}

module_path = PAGE_MAP[page]

import sys
# Deep purge of cached custom application modules to force full fresh load of changes
to_delete = [k for k in list(sys.modules.keys()) if k.startswith("src.")]
for k in to_delete:
    del sys.modules[k]

try:
    import importlib
    mod = importlib.import_module(module_path)
    importlib.reload(mod)
    mod.render()
except ModuleNotFoundError as exc:
    st.error(f"⚠️ Could not load page module **{module_path}**: `{exc}`")
    st.info("Make sure all frontend page files exist under `src/frontend/`.")
except Exception as exc:
    st.error(f"An error occurred while rendering **{page}**: `{exc}`")
    st.exception(exc)

