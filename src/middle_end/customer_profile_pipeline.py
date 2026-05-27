"""
Customer Profile Pipeline.

Creates user investment profiles and provides investor type classification
based on user inputs.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def create_user_profile(user_inputs):
    """
    Create a complete user investment profile from raw inputs.

    Args:
        user_inputs: dict with keys: age, income, risk_tolerance, investment_horizon,
                     preferred_sectors, portfolio_size, trading_frequency, past_returns

    Returns:
        dict with full profile including risk_score, investor_type, recommendations
    """
    try:
        from src.backend.customer_segmentation import (
            compute_risk_score, classify_investor_type,
            get_stock_recommendations, segment_user
        )

        result = segment_user(user_inputs)

        # Build comprehensive profile
        profile = {
            'user_inputs': user_inputs,
            'risk_score': result['risk_score'],
            'investor_type': result['segment'],
            'risk_level': result['risk_level'],
            'recommended_strategy': result['recommended_strategy'],
            'recommended_sectors': result['recommended_sectors'],
            'recommended_stocks': result.get('recommended_stocks', []),
            'investment_approach': result.get('investment_approach', ''),
            'risk_warning': result.get('risk_warning', ''),
            'profile_summary': _generate_profile_summary(user_inputs, result)
        }

        return profile

    except Exception as e:
        logger.error(f"Profile creation error: {e}")
        return _get_fallback_profile(user_inputs)


def _generate_profile_summary(user_inputs, result):
    """Generate a human-readable profile summary."""
    age = user_inputs.get('age', 'Unknown')
    horizon = user_inputs.get('investment_horizon', 'medium')
    segment = result['segment']
    risk_score = result['risk_score']

    summaries = {
        'Conservative': (
            f"Based on your profile (age {age}, risk score {risk_score}/10, "
            f"{horizon}-term horizon), you are classified as a Conservative investor. "
            f"We recommend focusing on stable, dividend-paying stocks and established "
            f"companies with consistent performance. Capital preservation should be "
            f"your primary goal."
        ),
        'Moderate': (
            f"Based on your profile (age {age}, risk score {risk_score}/10, "
            f"{horizon}-term horizon), you are classified as a Moderate investor. "
            f"We recommend a balanced portfolio mixing growth and value stocks "
            f"across multiple sectors. Consider a mix of established tech companies "
            f"and financial sector leaders."
        ),
        'Aggressive': (
            f"Based on your profile (age {age}, risk score {risk_score}/10, "
            f"{horizon}-term horizon), you are classified as an Aggressive investor. "
            f"You may consider high-growth stocks with greater volatility potential. "
            f"Focus on emerging tech, AI, and innovative companies, but be prepared "
            f"for significant price swings."
        )
    }

    return summaries.get(segment, summaries['Moderate'])


def _get_fallback_profile(user_inputs):
    """Generate a basic profile when backend is unavailable."""
    risk_tolerance = float(user_inputs.get('risk_tolerance', 5))

    if risk_tolerance <= 3:
        investor_type = 'Conservative'
    elif risk_tolerance <= 7:
        investor_type = 'Moderate'
    else:
        investor_type = 'Aggressive'

    return {
        'user_inputs': user_inputs,
        'risk_score': risk_tolerance,
        'investor_type': investor_type,
        'risk_level': 'Low' if risk_tolerance <= 3 else ('Medium' if risk_tolerance <= 7 else 'High'),
        'recommended_strategy': 'Diversify across sectors based on your risk profile.',
        'recommended_sectors': ['Technology', 'Healthcare', 'Finance'],
        'profile_summary': f'Based on your risk tolerance of {risk_tolerance}/10, '
                           f'you are classified as a {investor_type} investor.',
        'is_fallback': True
    }


def get_investor_description(investor_type):
    """
    Get a detailed description of an investor type.

    Args:
        investor_type: 'Conservative', 'Moderate', or 'Aggressive'

    Returns:
        str: detailed description
    """
    descriptions = {
        'Conservative': (
            "Conservative investors prioritize capital preservation and steady income. "
            "They prefer low-volatility investments like blue-chip stocks, bonds, and "
            "dividend-paying equities. These investors typically have a longer time "
            "horizon and lower risk tolerance. They accept lower potential returns "
            "in exchange for greater stability and predictability."
        ),
        'Moderate': (
            "Moderate investors seek a balance between growth and stability. They are "
            "willing to accept some volatility for the potential of higher returns. "
            "Their portfolios typically include a mix of growth stocks and value stocks, "
            "diversified across multiple sectors. They regularly rebalance their "
            "portfolios and maintain a medium-term investment horizon."
        ),
        'Aggressive': (
            "Aggressive investors seek maximum growth potential and are comfortable with "
            "significant price volatility. They focus on high-growth sectors like "
            "technology, biotech, and emerging markets. These investors typically have "
            "a higher risk tolerance, longer time horizon, and the financial capacity "
            "to absorb short-term losses in pursuit of long-term gains."
        )
    }

    return descriptions.get(investor_type, descriptions['Moderate'])


def get_risk_breakdown(user_inputs):
    """
    Get a detailed breakdown of how the risk score was computed.

    Args:
        user_inputs: dict with user profile data

    Returns:
        dict with component scores and weights
    """
    risk_tolerance = float(user_inputs.get('risk_tolerance', 5))

    freq_map = {'quarterly': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
    tf = user_inputs.get('trading_frequency', 'monthly')
    if isinstance(tf, str):
        tf_raw = freq_map.get(tf.lower(), 2)
    else:
        tf_raw = float(tf)
    tf_score = min(tf_raw / 4.0 * 10, 10)

    past_returns = float(user_inputs.get('past_returns', 0))
    returns_score = min(max((past_returns + 30) / 80 * 10, 0), 10)

    return {
        'risk_tolerance': {'value': risk_tolerance, 'weight': 0.4, 'weighted': round(risk_tolerance * 0.4, 2)},
        'trading_frequency': {'value': round(tf_score, 2), 'weight': 0.3, 'weighted': round(tf_score * 0.3, 2)},
        'past_returns': {'value': round(returns_score, 2), 'weight': 0.3, 'weighted': round(returns_score * 0.3, 2)},
        'total': round(risk_tolerance * 0.4 + tf_score * 0.3 + returns_score * 0.3, 2)
    }
