"""
Stock Feature Engineering Module
=================================
Creates technical indicators and target variables from raw OHLCV stock data.
All features are computed using pandas and numpy — no external TA libraries
are required.
"""

import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to a stock DataFrame.

    The following features are computed per ticker group:
        - daily_return:  (Close - previous Close) / previous Close
        - ma_7:          7-day simple moving average of Close
        - ma_30:         30-day simple moving average of Close
        - ma_100:        100-day simple moving average of Close
        - volatility:    20-day rolling standard deviation of daily_return
        - rsi:           14-day Relative Strength Index (manual calculation)
        - macd:          12-day EMA minus 26-day EMA of Close
        - macd_signal:   9-day EMA of MACD
        - volume_change: percentage change in Volume

    Args:
        df: Stock DataFrame with at least 'Ticker', 'Close', and 'Volume'
            columns.

    Returns:
        DataFrame with all technical feature columns appended.

    Raises:
        ValueError: If required columns ('Ticker', 'Close', 'Volume') are
                     missing.
    """
    required_cols = ['Ticker', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Helper to compute RSI for a single series
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI using the standard average-gain / average-loss method."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    # Process each ticker independently to avoid cross-contamination
    result_frames = []
    for _ticker, group in df.groupby('Ticker', sort=False):
        group = group.sort_values('Date').copy() if 'Date' in group.columns else group.copy()

        # Daily return
        group['daily_return'] = group['Close'].pct_change()

        # Moving averages
        group['ma_7'] = group['Close'].rolling(window=7, min_periods=1).mean()
        group['ma_14'] = group['Close'].rolling(window=14, min_periods=1).mean()
        group['ma_30'] = group['Close'].rolling(window=30, min_periods=1).mean()
        group['ma_50'] = group['Close'].rolling(window=50, min_periods=1).mean()

        # Volatility (20-day rolling std of daily return)
        group['volatility'] = group['daily_return'].rolling(window=20, min_periods=1).std()

        # RSI (14-day)
        group['rsi'] = _compute_rsi(group['Close'], period=14)

        # MACD
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema_12 - ema_26
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()

        # Volume change
        group['volume_change'] = group['Volume'].pct_change()

        # Price range
        group['price_range'] = (group['High'] - group['Low']) / group['Close']

        result_frames.append(group)

    df = pd.concat(result_frames, ignore_index=True)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target columns for supervised learning.

    Creates:
        - next_day_close: The closing price of the next trading day (shifted
          by -1 within each ticker group).
        - movement: Binary label — 1 if next_day_close > Close, else 0.

    Args:
        df: Stock DataFrame with at least 'Ticker' and 'Close' columns.

    Returns:
        DataFrame with 'next_day_close' and 'movement' columns appended.

    Raises:
        ValueError: If 'Ticker' or 'Close' columns are missing.
    """
    required_cols = ['Ticker', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    df['next_day_close'] = df.groupby('Ticker')['Close'].shift(-1)
    df['movement'] = (df['next_day_close'] > df['Close']).astype(int)

    return df


def get_feature_columns() -> list:
    """
    Return the canonical list of feature column names used for model training.

    Returns:
        A list of feature column name strings.
    """
    return [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume',
        'daily_return',
        'ma_7',
        'ma_14',
        'ma_30',
        'ma_50',
        'volatility',
        'rsi',
        'macd',
        'macd_signal',
        'volume_change',
        'price_range',
    ]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: add technical features, add targets, and drop NaN rows.

    This is the primary entry point for preparing a raw stock DataFrame for
    model training. It chains ``add_technical_features`` and ``add_targets``,
    then removes any rows containing NaN values introduced by rolling
    calculations or target shifting.

    Args:
        df: Raw stock DataFrame with at least 'Ticker', 'Date', 'Open',
            'High', 'Low', 'Close', and 'Volume' columns.

    Returns:
        A clean DataFrame ready for training, with all technical features
        and target columns, and no NaN values.

    Raises:
        ValueError: If required columns are missing.
    """
    df = add_technical_features(df)
    df = add_targets(df)

    # Drop rows with NaN introduced by rolling windows and target shift
    df = df.dropna().reset_index(drop=True)

    return df
