"""
Input Validation Utilities.

Validates user inputs for stock tickers, date ranges, user profiles,
and model selections.

Author: Stock Market AI Advisor Team
"""

import re
from datetime import datetime, timedelta


def validate_ticker(ticker):
    """
    Validate a stock ticker symbol.

    Args:
        ticker: string to validate

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not ticker or not isinstance(ticker, str):
        return False, "Ticker symbol is required."

    ticker = ticker.strip().upper()

    if len(ticker) < 1 or len(ticker) > 5:
        return False, "Ticker must be 1-5 characters long."

    if not re.match(r'^[A-Z]+$', ticker):
        return False, "Ticker must contain only uppercase letters."

    return True, ""


def validate_date_range(start_date, end_date):
    """
    Validate a date range for stock data queries.

    Args:
        start_date: start date (datetime or string)
        end_date: end date (datetime or string)

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return False, "Invalid date format. Use YYYY-MM-DD."

    if start_date >= end_date:
        return False, "Start date must be before end date."

    if end_date > datetime.now() + timedelta(days=1):
        return False, "End date cannot be in the future."

    if (end_date - start_date).days > 7300:  # ~20 years
        return False, "Date range too large. Maximum 20 years."

    if start_date < datetime(2000, 1, 1):
        return False, "Start date must be after 2000-01-01."

    return True, ""


def validate_user_inputs(inputs):
    """
    Validate user profile inputs for customer analytics.

    Args:
        inputs: dict with user profile fields

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    errors = []

    # Age
    age = inputs.get('age')
    if age is not None:
        try:
            age = int(age)
            if age < 18 or age > 100:
                errors.append("Age must be between 18 and 100.")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number.")

    # Income
    income = inputs.get('income')
    if income is not None:
        try:
            income = float(income)
            if income < 0:
                errors.append("Income cannot be negative.")
            if income > 100000000:
                errors.append("Income seems unrealistically high.")
        except (ValueError, TypeError):
            errors.append("Income must be a valid number.")

    # Risk tolerance
    risk = inputs.get('risk_tolerance')
    if risk is not None:
        try:
            risk = int(risk)
            if risk < 1 or risk > 10:
                errors.append("Risk tolerance must be between 1 and 10.")
        except (ValueError, TypeError):
            errors.append("Risk tolerance must be a valid number.")

    # Investment horizon
    horizon = inputs.get('investment_horizon')
    if horizon is not None:
        valid_horizons = ['short', 'medium', 'long', 'short-term', 'medium-term', 'long-term']
        if str(horizon).lower() not in valid_horizons:
            errors.append("Investment horizon must be 'short', 'medium', or 'long'.")

    # Portfolio size
    portfolio = inputs.get('portfolio_size')
    if portfolio is not None:
        try:
            portfolio = float(portfolio)
            if portfolio < 0:
                errors.append("Portfolio size cannot be negative.")
        except (ValueError, TypeError):
            errors.append("Portfolio size must be a valid number.")

    # Trading frequency
    freq = inputs.get('trading_frequency')
    if freq is not None:
        valid_freqs = ['daily', 'weekly', 'monthly', 'quarterly']
        if str(freq).lower() not in valid_freqs:
            errors.append("Trading frequency must be 'daily', 'weekly', 'monthly', or 'quarterly'.")

    # Past returns
    returns = inputs.get('past_returns')
    if returns is not None:
        try:
            returns = float(returns)
            if returns < -100 or returns > 1000:
                errors.append("Past returns percentage seems unrealistic (-100% to 1000%).")
        except (ValueError, TypeError):
            errors.append("Past returns must be a valid number.")

    if errors:
        return False, " | ".join(errors)

    return True, ""


def sanitize_text(text):
    """
    Clean and sanitize text input for sentiment analysis.

    Args:
        text: raw text string

    Returns:
        str: sanitized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Remove potentially harmful characters but keep meaningful punctuation
    text = re.sub(r'[<>{}|\\^~`]', '', text)

    # Limit length
    if len(text) > 5000:
        text = text[:5000]

    return text.strip()


def validate_model_name(model_name):
    """
    Validate that a model name is one of the supported options.

    Args:
        model_name: string model identifier

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    valid_models = [
        'random_forest', 'xgboost', 'lstm',
        'random_forest_regressor', 'random_forest_classifier',
        'xgboost_regressor', 'xgboost_classifier'
    ]

    if not model_name or not isinstance(model_name, str):
        return False, "Model name is required."

    if model_name.lower() not in valid_models:
        return False, f"Invalid model. Choose from: {', '.join(valid_models[:3])}"

    return True, ""
