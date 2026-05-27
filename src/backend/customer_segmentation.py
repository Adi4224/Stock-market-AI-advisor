"""
Customer Segmentation for Investment Recommendations.

Segments users into Conservative, Moderate, and Aggressive investor profiles
using KMeans clustering and rule-based scoring, then suggests appropriate
stock categories for each segment.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
PLOTS_DIR = os.path.join(REPORTS_DIR, 'training_plots')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

for d in [MODEL_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger(__name__)

# Trading frequency encoding
FREQUENCY_MAP = {'quarterly': 1, 'monthly': 2, 'weekly': 3, 'daily': 4}
HORIZON_MAP = {'short': 1, 'medium': 2, 'long': 3}


def compute_risk_score(user_data):
    """
    Compute a risk score (0-10) based on user attributes.

    Args:
        user_data: dict with keys: risk_tolerance, trading_frequency, past_returns,
                   and optionally age, portfolio_size

    Returns:
        float: risk score on 0-10 scale
    """
    risk_tolerance = float(user_data.get('risk_tolerance', 5))

    # Normalize trading frequency
    tf = user_data.get('trading_frequency', 'monthly')
    if isinstance(tf, str):
        tf_score = FREQUENCY_MAP.get(tf.lower(), 2) / 4.0 * 10
    else:
        tf_score = min(float(tf) / 4.0 * 10, 10)

    # Normalize past returns (-30 to 50 -> 0 to 10)
    past_returns = float(user_data.get('past_returns', 0))
    returns_score = min(max((past_returns + 30) / 80 * 10, 0), 10)

    # Weighted combination
    risk_score = (risk_tolerance * 0.4 + tf_score * 0.3 + returns_score * 0.3)

    # Age adjustment (younger = slightly higher risk appetite)
    age = user_data.get('age', None)
    if age is not None:
        age = float(age)
        if age < 30:
            risk_score *= 1.05
        elif age > 55:
            risk_score *= 0.90

    return round(min(max(risk_score, 0), 10), 2)


def classify_investor_type(risk_score):
    """
    Classify an investor based on their risk score.

    Args:
        risk_score: float between 0 and 10

    Returns:
        str: 'Conservative', 'Moderate', or 'Aggressive'
    """
    if risk_score <= 3.5:
        return 'Conservative'
    elif risk_score <= 6.5:
        return 'Moderate'
    else:
        return 'Aggressive'


def train_kmeans_segmentation(user_df, n_clusters=3):
    """
    Train a KMeans clustering model on user behavior data.

    Args:
        user_df: DataFrame with user attributes
        n_clusters: number of clusters (default 3)

    Returns:
        tuple: (trained KMeans model, label_map dict, scaler)
    """
    # Prepare features for clustering
    feature_cols = []
    df = user_df.copy()

    # Encode categorical features
    if 'trading_frequency' in df.columns:
        df['tf_encoded'] = df['trading_frequency'].str.lower().map(FREQUENCY_MAP).fillna(2)
        feature_cols.append('tf_encoded')

    if 'investment_horizon' in df.columns:
        df['ih_encoded'] = df['investment_horizon'].str.lower().map(HORIZON_MAP).fillna(2)
        feature_cols.append('ih_encoded')

    for col in ['risk_tolerance', 'past_returns', 'portfolio_size', 'age', 'income']:
        if col in df.columns:
            feature_cols.append(col)

    if not feature_cols:
        logger.error("No suitable features for clustering")
        return None, None, None

    X = df[feature_cols].fillna(df[feature_cols].median()).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Map cluster labels to investor types based on centroid risk levels
    risk_col_idx = feature_cols.index('risk_tolerance') if 'risk_tolerance' in feature_cols else 0
    centroid_risks = kmeans.cluster_centers_[:, risk_col_idx]
    sorted_indices = np.argsort(centroid_risks)

    label_map = {}
    investor_types = ['Conservative', 'Moderate', 'Aggressive']
    for i, idx in enumerate(sorted_indices):
        label_map[int(idx)] = investor_types[min(i, 2)]

    # Save model
    model_data = {
        'kmeans': kmeans,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'label_map': label_map
    }

    model_path = os.path.join(MODEL_DIR, 'user_segment_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"KMeans segmentation model saved to {model_path}")
    logger.info(f"Label mapping: {label_map}")

    return kmeans, label_map, scaler


def segment_user(user_data):
    """
    Segment a single user based on their profile data.

    Uses rule-based scoring (always available, no model needed).

    Args:
        user_data: dict with user profile attributes

    Returns:
        dict with segment, risk_score, risk_level, recommended_sectors, recommended_strategy
    """
    risk_score = compute_risk_score(user_data)
    investor_type = classify_investor_type(risk_score)
    recommendations = get_stock_recommendations(investor_type)

    # Risk level description
    if risk_score <= 2:
        risk_level = 'Very Low'
    elif risk_score <= 4:
        risk_level = 'Low'
    elif risk_score <= 6:
        risk_level = 'Medium'
    elif risk_score <= 8:
        risk_level = 'High'
    else:
        risk_level = 'Very High'

    return {
        'segment': investor_type,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommended_sectors': recommendations['sectors'],
        'recommended_strategy': recommendations['strategy'],
        'recommended_stocks': recommendations['stocks'],
        'investment_approach': recommendations['approach'],
        'risk_warning': recommendations['risk_warning']
    }


def get_stock_recommendations(segment):
    """
    Get stock recommendations for a given investor segment.

    Args:
        segment: str - 'Conservative', 'Moderate', or 'Aggressive'

    Returns:
        dict with stocks, sectors, strategy, approach, and risk_warning
    """
    recommendations = {
        'Conservative': {
            'stocks': [
                {'ticker': 'JNJ', 'company': 'Johnson & Johnson', 'sector': 'Healthcare'},
                {'ticker': 'PG', 'company': 'Procter & Gamble', 'sector': 'Consumer'},
                {'ticker': 'WMT', 'company': 'Walmart', 'sector': 'Consumer'},
                {'ticker': 'V', 'company': 'Visa', 'sector': 'Finance'},
                {'ticker': 'UNH', 'company': 'UnitedHealth', 'sector': 'Healthcare'},
                {'ticker': 'KO', 'company': 'Coca-Cola', 'sector': 'Consumer'},
            ],
            'sectors': ['Healthcare', 'Consumer Staples', 'Utilities', 'Finance'],
            'strategy': 'Focus on blue-chip stocks with stable dividends and low volatility. '
                        'Prioritize capital preservation over growth.',
            'approach': 'Long-term buy-and-hold with dividend reinvestment',
            'risk_warning': 'Even conservative investments carry market risk.'
        },
        'Moderate': {
            'stocks': [
                {'ticker': 'AAPL', 'company': 'Apple Inc.', 'sector': 'Technology'},
                {'ticker': 'MSFT', 'company': 'Microsoft', 'sector': 'Technology'},
                {'ticker': 'GOOGL', 'company': 'Alphabet', 'sector': 'Technology'},
                {'ticker': 'JPM', 'company': 'JPMorgan Chase', 'sector': 'Finance'},
                {'ticker': 'HD', 'company': 'Home Depot', 'sector': 'Consumer'},
                {'ticker': 'MA', 'company': 'Mastercard', 'sector': 'Finance'},
            ],
            'sectors': ['Technology', 'Finance', 'Consumer Discretionary', 'Healthcare'],
            'strategy': 'Balance growth and value stocks. Diversify across sectors '
                        'with moderate risk exposure.',
            'approach': 'Mix of growth and value investing with regular rebalancing',
            'risk_warning': 'Moderate risk investments may experience significant short-term volatility.'
        },
        'Aggressive': {
            'stocks': [
                {'ticker': 'TSLA', 'company': 'Tesla', 'sector': 'Technology'},
                {'ticker': 'NVDA', 'company': 'NVIDIA', 'sector': 'Technology'},
                {'ticker': 'AMZN', 'company': 'Amazon', 'sector': 'Technology'},
                {'ticker': 'META', 'company': 'Meta Platforms', 'sector': 'Technology'},
                {'ticker': 'AMD', 'company': 'AMD', 'sector': 'Technology'},
                {'ticker': 'NFLX', 'company': 'Netflix', 'sector': 'Entertainment'},
            ],
            'sectors': ['Technology', 'AI/ML', 'Electric Vehicles', 'Crypto-adjacent'],
            'strategy': 'Focus on high-growth tech stocks with strong momentum. '
                        'Accept higher volatility for potentially higher returns.',
            'approach': 'Active trading with momentum-based strategies',
            'risk_warning': 'Aggressive investments carry high risk of significant losses.'
        }
    }

    return recommendations.get(segment, recommendations['Moderate'])


def plot_user_segments(user_df, labels, save_path=None):
    """
    Plot user segments using PCA for 2D visualization.

    Args:
        user_df: DataFrame with user features
        labels: cluster labels
        save_path: optional path to save the figure
    """
    feature_cols = ['risk_tolerance', 'past_returns']
    for col in ['portfolio_size', 'age', 'income']:
        if col in user_df.columns:
            feature_cols.append(col)

    available = [c for c in feature_cols if c in user_df.columns]
    if len(available) < 2:
        logger.warning("Not enough features for PCA visualization")
        return

    X = user_df[available].fillna(user_df[available].median()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    colors = {'Conservative': '#00D4FF', 'Moderate': '#FFD700', 'Aggressive': '#FF5252'}

    fig, ax = plt.subplots(figsize=(10, 7))

    for segment, color in colors.items():
        mask = labels == segment
        if mask.any():
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=segment,
                       alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    ax.set_title('User Segmentation (PCA Projection)', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Demo
    print("Customer Segmentation Demo")
    print("=" * 60)

    test_users = [
        {'age': 60, 'income': 80000, 'risk_tolerance': 2, 'investment_horizon': 'long',
         'trading_frequency': 'quarterly', 'past_returns': 5, 'portfolio_size': 200000},
        {'age': 35, 'income': 120000, 'risk_tolerance': 5, 'investment_horizon': 'medium',
         'trading_frequency': 'weekly', 'past_returns': 12, 'portfolio_size': 50000},
        {'age': 25, 'income': 90000, 'risk_tolerance': 9, 'investment_horizon': 'short',
         'trading_frequency': 'daily', 'past_returns': 30, 'portfolio_size': 15000},
    ]

    for i, user in enumerate(test_users):
        result = segment_user(user)
        print(f"\nUser {i + 1}: {result['segment']} (Risk Score: {result['risk_score']})")
        print(f"  Strategy: {result['recommended_strategy']}")
        print(f"  Sectors: {result['recommended_sectors']}")
