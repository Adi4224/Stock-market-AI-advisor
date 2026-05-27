"""
Stock Suggestion Engine.

Suggests stocks based on user investment profile, risk level, and preferences.
Includes explanations and financial disclaimers.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DISCLAIMER = ("⚠️ DISCLAIMER: These stock suggestions are for educational purposes only "
              "and do NOT constitute financial advice. Past performance does not guarantee "
              "future results. Always consult a qualified financial advisor before making "
              "investment decisions.")

# Comprehensive stock database
STOCK_DATABASE = {
    # ── Technology ────────────────────────────────────────────────────────────
    'AAPL':  {'company': 'Apple Inc.',            'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'MSFT':  {'company': 'Microsoft Corp.',       'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'GOOGL': {'company': 'Alphabet Inc.',         'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'AMZN':  {'company': 'Amazon.com Inc.',       'sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Mega'},
    'META':  {'company': 'Meta Platforms',        'sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Mega'},
    'NVDA':  {'company': 'NVIDIA Corp.',          'sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Mega'},
    'AMD':   {'company': 'Advanced Micro Devices','sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Large'},
    'INTC':  {'company': 'Intel Corp.',           'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Large'},
    'CRM':   {'company': 'Salesforce Inc.',       'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Large'},
    'ADBE':  {'company': 'Adobe Inc.',            'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Large'},
    'ORCL':  {'company': 'Oracle Corp.',          'sector': 'Technology',      'risk_level': 'Moderate',     'market_cap': 'Large'},
    'NFLX':  {'company': 'Netflix Inc.',          'sector': 'Entertainment',   'risk_level': 'Aggressive',   'market_cap': 'Large'},
    'PYPL':  {'company': 'PayPal Holdings',       'sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Large'},
    'UBER':  {'company': 'Uber Technologies',     'sector': 'Technology',      'risk_level': 'Aggressive',   'market_cap': 'Large'},
    # ── Automotive / Tech ─────────────────────────────────────────────────────
    'TSLA':  {'company': 'Tesla Inc.',            'sector': 'Automotive/Tech', 'risk_level': 'Aggressive',   'market_cap': 'Mega'},
    # ── Finance ───────────────────────────────────────────────────────────────
    'JPM':   {'company': 'JPMorgan Chase',        'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'BAC':   {'company': 'Bank of America',       'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'GS':    {'company': 'Goldman Sachs',         'sector': 'Finance',         'risk_level': 'Aggressive',   'market_cap': 'Large'},
    'MS':    {'company': 'Morgan Stanley',        'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'WFC':   {'company': 'Wells Fargo',           'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'V':     {'company': 'Visa Inc.',             'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'MA':    {'company': 'Mastercard Inc.',       'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'AXP':   {'company': 'American Express',      'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'C':     {'company': 'Citigroup Inc.',        'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    'BLK':   {'company': 'BlackRock Inc.',        'sector': 'Finance',         'risk_level': 'Moderate',     'market_cap': 'Large'},
    # ── Healthcare ────────────────────────────────────────────────────────────
    'JNJ':   {'company': 'Johnson & Johnson',     'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Mega'},
    'PFE':   {'company': 'Pfizer Inc.',           'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Large'},
    'UNH':   {'company': 'UnitedHealth Group',    'sector': 'Healthcare',      'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'MRK':   {'company': 'Merck & Co.',           'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Large'},
    'ABT':   {'company': 'Abbott Laboratories',   'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Large'},
    'LLY':   {'company': 'Eli Lilly & Co.',       'sector': 'Healthcare',      'risk_level': 'Moderate',     'market_cap': 'Mega'},
    'ABBV':  {'company': 'AbbVie Inc.',           'sector': 'Healthcare',      'risk_level': 'Moderate',     'market_cap': 'Large'},
    'TMO':   {'company': 'Thermo Fisher Scientific','sector': 'Healthcare',    'risk_level': 'Moderate',     'market_cap': 'Large'},
    'MDT':   {'company': 'Medtronic plc',         'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Large'},
    'BMY':   {'company': 'Bristol-Myers Squibb',  'sector': 'Healthcare',      'risk_level': 'Conservative', 'market_cap': 'Large'},
    # ── Energy ────────────────────────────────────────────────────────────────
    'XOM':   {'company': 'Exxon Mobil',           'sector': 'Energy',          'risk_level': 'Moderate',     'market_cap': 'Large'},
    'CVX':   {'company': 'Chevron Corp.',         'sector': 'Energy',          'risk_level': 'Moderate',     'market_cap': 'Large'},
    'COP':   {'company': 'ConocoPhillips',        'sector': 'Energy',          'risk_level': 'Moderate',     'market_cap': 'Large'},
    'SLB':   {'company': 'Schlumberger Ltd.',     'sector': 'Energy',          'risk_level': 'Aggressive',   'market_cap': 'Large'},
    'EOG':   {'company': 'EOG Resources',         'sector': 'Energy',          'risk_level': 'Aggressive',   'market_cap': 'Large'},
    # ── Consumer ──────────────────────────────────────────────────────────────
    'WMT':   {'company': 'Walmart Inc.',          'sector': 'Consumer',        'risk_level': 'Conservative', 'market_cap': 'Mega'},
    'PG':    {'company': 'Procter & Gamble',      'sector': 'Consumer',        'risk_level': 'Conservative', 'market_cap': 'Large'},
    'KO':    {'company': 'Coca-Cola Co.',         'sector': 'Consumer',        'risk_level': 'Conservative', 'market_cap': 'Large'},
    'PEP':   {'company': 'PepsiCo Inc.',          'sector': 'Consumer',        'risk_level': 'Conservative', 'market_cap': 'Large'},
    'COST':  {'company': 'Costco Wholesale',      'sector': 'Consumer',        'risk_level': 'Moderate',     'market_cap': 'Large'},
    'HD':    {'company': 'Home Depot',            'sector': 'Consumer',        'risk_level': 'Moderate',     'market_cap': 'Large'},
    'NKE':   {'company': 'Nike Inc.',             'sector': 'Consumer',        'risk_level': 'Moderate',     'market_cap': 'Large'},
    'MCD':   {'company': "McDonald's Corp.",      'sector': 'Consumer',        'risk_level': 'Conservative', 'market_cap': 'Large'},
    'SBUX':  {'company': 'Starbucks Corp.',       'sector': 'Consumer',        'risk_level': 'Moderate',     'market_cap': 'Large'},
    # ── Entertainment / Communication ─────────────────────────────────────────
    'DIS':   {'company': 'Walt Disney Co.',       'sector': 'Entertainment',   'risk_level': 'Moderate',     'market_cap': 'Large'},
}

SUGGESTION_REASONS = {
    'Conservative': {
        'general': 'Stable blue-chip company with consistent dividends and low volatility.',
        'Healthcare': 'Healthcare sector provides defensive positioning during market downturns.',
        'Consumer': 'Consumer staples offer recession-resistant revenue streams.',
        'Finance': 'Established financial institution with strong regulatory standing.',
        'Energy': 'Major energy company with strong cash flow and dividend yield.',
    },
    'Moderate': {
        'general': 'Strong growth potential with manageable risk and solid fundamentals.',
        'Technology': 'Leading tech company with diversified revenue and market dominance.',
        'Finance': 'Financial leader with growth exposure and attractive valuations.',
        'Consumer': 'Consumer discretionary with strong brand value and growth trajectory.',
        'Healthcare': 'Healthcare innovator with pipeline upside and stable revenue base.',
        'Energy': 'Integrated energy company balancing growth with shareholder returns.',
    },
    'Aggressive': {
        'general': 'High-growth company with significant upside potential and innovation focus.',
        'Technology': 'Cutting-edge tech with AI/ML exposure and rapid revenue growth.',
        'Automotive/Tech': 'Disruptive player in EV and autonomy with high-risk/high-reward profile.',
        'Entertainment': 'Digital-first entertainment with subscriber growth momentum.',
        'Finance': 'Investment bank with high earnings leverage and market sensitivity.',
        'Energy': 'Energy services or E&P company with cyclical upside and commodity exposure.',
    }
}


def suggest_stocks(user_profile):
    """
    Suggest stocks based on user investment profile.

    Args:
        user_profile: dict with investor_type, risk_score, recommended_sectors, etc.

    Returns:
        list of suggestion dicts with ticker, company, sector, risk_level, reason
    """
    investor_type = user_profile.get('investor_type', 'Moderate')
    preferred_sectors = user_profile.get('recommended_sectors', [])

    # Select stocks matching investor type
    risk_mapping = {
        'Conservative': ['Conservative'],
        'Moderate': ['Conservative', 'Moderate'],
        'Aggressive': ['Moderate', 'Aggressive']
    }

    allowed_risks = risk_mapping.get(investor_type, ['Moderate'])

    # Filter stocks
    candidates = []
    for ticker, info in STOCK_DATABASE.items():
        if info['risk_level'] in allowed_risks:
            score = _compute_match_score(info, investor_type, preferred_sectors)
            candidates.append({
                'ticker': ticker,
                'company': info['company'],
                'sector': info['sector'],
                'risk_level': info['risk_level'],
                'market_cap': info['market_cap'],
                'match_score': score,
                'reason': _get_reason(investor_type, info['sector']),
                'expected_volatility': _get_volatility(info['risk_level'])
            })

    # Sort by match score and return top suggestions
    candidates.sort(key=lambda x: x['match_score'], reverse=True)
    suggestions = candidates[:8]

    # Add disclaimer to each
    for s in suggestions:
        s['disclaimer'] = DISCLAIMER

    return suggestions


def _compute_match_score(stock_info, investor_type, preferred_sectors):
    """Compute a match score for a stock based on user preferences."""
    score = 50  # Base score

    # Risk alignment
    if investor_type == 'Conservative' and stock_info['risk_level'] == 'Conservative':
        score += 30
    elif investor_type == 'Moderate' and stock_info['risk_level'] == 'Moderate':
        score += 30
    elif investor_type == 'Aggressive' and stock_info['risk_level'] == 'Aggressive':
        score += 30
    else:
        score += 10

    # Sector preference
    if preferred_sectors:
        for sector in preferred_sectors:
            if sector.lower() in stock_info['sector'].lower():
                score += 20
                break

    # Market cap bonus
    if stock_info['market_cap'] == 'Mega':
        score += 5

    return score


def _get_reason(investor_type, sector):
    """Get a human-readable reason for the stock suggestion."""
    type_reasons = SUGGESTION_REASONS.get(investor_type, SUGGESTION_REASONS['Moderate'])
    return type_reasons.get(sector, type_reasons['general'])


def _get_volatility(risk_level):
    """Map risk level to expected volatility description."""
    return {
        'Conservative': 'Low (< 15% annual)',
        'Moderate': 'Medium (15-30% annual)',
        'Aggressive': 'High (> 30% annual)'
    }.get(risk_level, 'Medium')


def get_sector_stocks(sector):
    """Get all stocks in a specific sector."""
    return [
        {'ticker': t, **info}
        for t, info in STOCK_DATABASE.items()
        if sector.lower() in info['sector'].lower()
    ]


def get_risk_matched_stocks(risk_level):
    """Get all stocks matching a specific risk level."""
    return [
        {'ticker': t, **info}
        for t, info in STOCK_DATABASE.items()
        if info['risk_level'] == risk_level
    ]


def explain_suggestion(suggestion):
    """Generate a detailed explanation for a stock suggestion."""
    return (
        f"📊 {suggestion['ticker']} ({suggestion['company']})\n"
        f"Sector: {suggestion['sector']} | Risk: {suggestion['risk_level']}\n"
        f"Expected Volatility: {suggestion['expected_volatility']}\n"
        f"Reason: {suggestion['reason']}\n\n"
        f"{DISCLAIMER}"
    )
