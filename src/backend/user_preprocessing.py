"""
User Behavior Data Preprocessing Module
=========================================
Provides functions for cleaning, encoding, and engineering features from user
behavior and profile data. Computes composite scores (risk, behavior,
investment preference) used for personalized stock recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocess_user_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess user behavior and profile data for downstream modelling.

    Performs the following steps:
        1. Handles missing values: median imputation for numeric columns,
           mode imputation for categorical columns.
        2. Encodes ordinal columns:
           - investment_horizon: short=1, medium=2, long=3
           - trading_frequency: quarterly=1, monthly=2, weekly=3, daily=4
        3. Label-encodes the 'preferred_sectors' column.
        4. Computes composite scores:
           - risk_score: weighted combination of risk_tolerance,
             trading_frequency_encoded, and past_returns.
           - behavior_score: combination of trading_frequency_encoded,
             portfolio_size (normalized), and investment_horizon_encoded.
           - investment_preference_score: combination of income (normalized),
             portfolio_size (normalized), and risk_tolerance.
        5. Scales all numeric features using MinMaxScaler.

    Args:
        df: Raw user DataFrame. Expected columns include (but are not limited
            to): 'risk_tolerance', 'trading_frequency', 'investment_horizon',
            'preferred_sectors', 'past_returns', 'portfolio_size', 'income'.

    Returns:
        Processed DataFrame with encoded, engineered, and scaled features.

    Raises:
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Handle missing values
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_value = mode_val.iloc[0] if not mode_val.empty else 'unknown'
            df[col] = df[col].fillna(fill_value)

    # ------------------------------------------------------------------
    # 2. Encode ordinal columns
    # ------------------------------------------------------------------
    horizon_map = {'short': 1, 'medium': 2, 'long': 3}
    if 'investment_horizon' in df.columns:
        df['investment_horizon_encoded'] = (
            df['investment_horizon']
            .str.lower()
            .map(horizon_map)
            .fillna(2)  # default to medium if unrecognized
            .astype(int)
        )

    frequency_map = {'quarterly': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
    if 'trading_frequency' in df.columns:
        df['trading_frequency_encoded'] = (
            df['trading_frequency']
            .str.lower()
            .map(frequency_map)
            .fillna(1)  # default to quarterly if unrecognized
            .astype(int)
        )

    # ------------------------------------------------------------------
    # 3. Label-encode preferred_sectors
    # ------------------------------------------------------------------
    if 'preferred_sectors' in df.columns:
        le = LabelEncoder()
        df['preferred_sectors_encoded'] = le.fit_transform(
            df['preferred_sectors'].astype(str)
        )

    # ------------------------------------------------------------------
    # 4. Compute composite scores
    # ------------------------------------------------------------------
    # Normalize helper columns for score computation
    if 'portfolio_size' in df.columns:
        ps_min = df['portfolio_size'].min()
        ps_max = df['portfolio_size'].max()
        df['portfolio_size_normalized'] = (
            (df['portfolio_size'] - ps_min) / (ps_max - ps_min)
            if ps_max != ps_min
            else 0.0
        )
    else:
        df['portfolio_size_normalized'] = 0.0

    if 'income' in df.columns:
        inc_min = df['income'].min()
        inc_max = df['income'].max()
        df['income_normalized'] = (
            (df['income'] - inc_min) / (inc_max - inc_min)
            if inc_max != inc_min
            else 0.0
        )
    else:
        df['income_normalized'] = 0.0

    # --- risk_score ---
    risk_tolerance = df.get('risk_tolerance', pd.Series(0.0, index=df.index))
    trading_freq_enc = df.get('trading_frequency_encoded', pd.Series(0.0, index=df.index))
    past_returns = df.get('past_returns', pd.Series(0.0, index=df.index))

    df['risk_score'] = (
        0.5 * risk_tolerance
        + 0.3 * trading_freq_enc
        + 0.2 * past_returns
    )

    # --- behavior_score ---
    horizon_enc = df.get('investment_horizon_encoded', pd.Series(0.0, index=df.index))

    df['behavior_score'] = (
        0.4 * trading_freq_enc
        + 0.35 * df['portfolio_size_normalized']
        + 0.25 * horizon_enc
    )

    # --- investment_preference_score ---
    df['investment_preference_score'] = (
        0.4 * df['income_normalized']
        + 0.35 * df['portfolio_size_normalized']
        + 0.25 * risk_tolerance
    )

    # ------------------------------------------------------------------
    # 5. Scale numeric features
    # ------------------------------------------------------------------
    # Re-detect numeric columns after engineering
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_features:
        scaler = MinMaxScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df
