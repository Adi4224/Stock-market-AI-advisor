"""
Result Formatter.

Formats prediction, sentiment, and suggestion results for display
in the Streamlit frontend.

Author: Stock Market AI Advisor Team
"""


def format_prediction_result(prediction):
    """
    Format a stock prediction result for display.

    Args:
        prediction: dict from prediction_pipeline

    Returns:
        dict with added 'display' sub-dict containing formatted values
    """
    result = prediction.copy()

    price = prediction.get('predicted_price', 0)
    current = prediction.get('current_price', 0)
    change = prediction.get('change_pct', 0)

    result['display'] = {
        'predicted_price': format_currency(price),
        'current_price': format_currency(current),
        'change_pct': format_percentage(change),
        'change_color': '#00C853' if change >= 0 else '#FF5252',
        'change_icon': '📈' if change >= 0 else '📉',
        'model_badge': prediction.get('model_used', 'Unknown'),
        'confidence_badge': prediction.get('confidence', 'N/A'),
        'is_demo_badge': '🔬 Demo Mode' if prediction.get('is_demo', False) else '✅ Live Model',
    }

    return result


def format_sentiment_result(sentiment):
    """
    Format a sentiment analysis result for display.

    Args:
        sentiment: dict from sentiment_pipeline

    Returns:
        dict with added display formatting
    """
    result = sentiment.copy()

    label = sentiment.get('sentiment_label', 'Neutral')
    score = sentiment.get('sentiment_score', 0)

    emoji_map = {'Positive': '😊', 'Negative': '😟', 'Neutral': '😐'}
    color_map = {'Positive': '#00C853', 'Negative': '#FF5252', 'Neutral': '#FFD700'}

    result['display'] = {
        'emoji': emoji_map.get(label, '😐'),
        'color': color_map.get(label, '#FFD700'),
        'score_formatted': f"{score:+.4f}",
        'label_badge': f"{emoji_map.get(label, '')} {label}",
        'gauge_value': (score + 1) / 2 * 100,  # Convert -1..1 to 0..100
        'confidence_pct': f"{abs(score) * 100:.1f}%"
    }

    return result


def format_suggestion_result(suggestions):
    """
    Format stock suggestion results for display.

    Args:
        suggestions: list of suggestion dicts

    Returns:
        list of formatted suggestion dicts
    """
    formatted = []
    risk_colors = {
        'Conservative': '#00D4FF',
        'Moderate': '#FFD700',
        'Aggressive': '#FF5252'
    }

    risk_icons = {
        'Conservative': '🛡️',
        'Moderate': '⚖️',
        'Aggressive': '🚀'
    }

    for s in suggestions:
        fs = s.copy()
        risk = s.get('risk_level', 'Moderate')
        fs['display'] = {
            'risk_color': risk_colors.get(risk, '#FFD700'),
            'risk_icon': risk_icons.get(risk, '⚖️'),
            'risk_badge': f"{risk_icons.get(risk, '')} {risk}",
            'sector_badge': s.get('sector', 'Unknown'),
            'title': f"{s.get('ticker', '')} - {s.get('company', '')}",
        }
        formatted.append(fs)

    return formatted


def format_metrics_table(metrics):
    """
    Format model evaluation metrics for display in a table.

    Args:
        metrics: dict or list of dicts with metric values

    Returns:
        pd.DataFrame formatted for display
    """
    import pandas as pd

    if isinstance(metrics, dict):
        metrics = [metrics]

    df = pd.DataFrame(metrics)

    # Format numeric columns
    for col in ['mae', 'mse', 'rmse']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    for col in ['r2', 'accuracy', 'precision', 'recall', 'f1']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    return df


def get_disclaimer_text():
    """Get the standard financial disclaimer text."""
    return (
        "⚠️ **DISCLAIMER**: This application is for educational purposes only and does not "
        "provide financial advice. All predictions are model-based estimates and should not "
        "be used as the sole basis for investment decisions. Past performance does not "
        "guarantee future results. Always consult a qualified financial advisor before "
        "making investment decisions."
    )


def format_currency(value, convert=True):
    """Format a number as currency. Supports USD and INR conversion via Streamlit state."""
    try:
        import streamlit as st
        currency = st.session_state.get("currency", "USD")
    except Exception:
        currency = "USD"

    try:
        value = float(value)
        if currency == "INR":
            if convert:
                value *= 83.5
            return f"₹{value:,.2f}"
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        if currency == "INR":
            return "₹0.00"
        return "$0.00"


def format_percentage(value):
    """Format a number as percentage. E.g., 12.34 -> '+12.34%'"""
    try:
        value = float(value)
        sign = '+' if value >= 0 else ''
        return f"{sign}{value:.2f}%"
    except (ValueError, TypeError):
        return "0.00%"


def format_large_number(value):
    """Format a large number. E.g., 1200000 -> '1.2M'"""
    try:
        value = float(value)
        if abs(value) >= 1e12:
            return f"{value / 1e12:.1f}T"
        elif abs(value) >= 1e9:
            return f"{value / 1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value / 1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value / 1e3:.1f}K"
        else:
            return f"{value:,.0f}"
    except (ValueError, TypeError):
        return "0"
