"""
Stock Data Preprocessing Module
================================
Provides functions for cleaning, preprocessing, and preparing stock market data
for machine learning models. Handles missing values, duplicates, outliers,
train/test splitting, and feature scaling.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw stock market data.

    Performs the following steps:
        1. Ensures the 'Date' column is in datetime format.
        2. Sorts the DataFrame by 'Ticker' and 'Date'.
        3. Removes duplicate rows based on 'Ticker' and 'Date'.
        4. Forward-fills missing values for OHLCV columns within each ticker group.
        5. Caps outliers using the IQR method (values are capped, not removed).

    Args:
        df: Raw stock DataFrame with columns including
            'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns:
        Cleaned and preprocessed DataFrame.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Sort by Ticker and Date
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # Remove duplicate rows (same Ticker + Date)
    df = df.drop_duplicates(subset=['Ticker', 'Date'], keep='first').reset_index(drop=True)

    # Forward-fill missing OHLCV values within each ticker group
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[ohlcv_cols] = df.groupby('Ticker')[ohlcv_cols].transform(
        lambda group: group.ffill()
    )

    # Handle outliers using IQR method (cap, don't remove)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def get_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    test_size: float = 0.2,
    time_based: bool = True,
) -> tuple:
    """
    Split stock data into training and testing sets.

    Args:
        df: Preprocessed stock DataFrame.
        target_col: Name of the target column.
        feature_cols: List of feature column names.
        test_size: Proportion of data to use as the test set (default 0.2).
        time_based: If True, performs a chronological split without shuffling
                     to prevent data leakage. If False, performs a random split.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If target_col or any feature_col is missing from the DataFrame.
    """
    missing = [c for c in [target_col] + feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    X = df[feature_cols].values
    y = df[target_col].values

    if time_based:
        # Chronological split — no shuffling
        split_index = int(len(df) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str = None,
) -> tuple:
    """
    Scale features using StandardScaler.

    Args:
        X_train: Training feature array.
        X_test: Testing feature array.
        scaler_path: Optional file path to save the fitted scaler as a pickle
                      file. If None, the scaler is not persisted to disk.

    Returns:
        A tuple of (X_train_scaled, X_test_scaled, scaler).

    Raises:
        IOError: If the scaler cannot be saved to the specified path.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path is not None:
        try:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        except Exception as e:
            raise IOError(f"Failed to save scaler to '{scaler_path}': {e}")

    return X_train_scaled, X_test_scaled, scaler
