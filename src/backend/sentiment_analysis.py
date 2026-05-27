"""
Financial News Sentiment Analysis using VADER.

Analyzes financial news headlines and articles for sentiment
(Positive, Negative, Neutral) using the VADER sentiment analyzer.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def get_vader_analyzer():
    """
    Initialize and return the VADER sentiment analyzer.

    Returns:
        SentimentIntensityAnalyzer instance
    """
    return SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment of a single text string.

    Args:
        text: string to analyze

    Returns:
        dict with keys: text, compound_score, pos, neg, neu, label
    """
    if not text or not isinstance(text, str):
        return {
            'text': str(text),
            'compound_score': 0.0,
            'pos': 0.0,
            'neg': 0.0,
            'neu': 1.0,
            'label': 'Neutral'
        }

    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(text)

    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'

    return {
        'text': text,
        'compound_score': round(compound, 4),
        'pos': round(scores['pos'], 4),
        'neg': round(scores['neg'], 4),
        'neu': round(scores['neu'], 4),
        'label': label
    }


def analyze_batch(texts):
    """
    Analyze sentiment for a list of texts.

    Args:
        texts: list of strings to analyze

    Returns:
        pd.DataFrame with columns: text, compound_score, pos, neg, neu, label
    """
    results = [analyze_sentiment(text) for text in texts]
    return pd.DataFrame(results)


def analyze_news_dataframe(df, text_col='headline'):
    """
    Add sentiment columns to an existing DataFrame.

    Args:
        df: DataFrame with a text column
        text_col: name of the column containing text to analyze

    Returns:
        DataFrame with added sentiment columns
    """
    df = df.copy()

    if text_col not in df.columns:
        logger.warning(f"Column '{text_col}' not found in DataFrame")
        return df

    analyzer = get_vader_analyzer()

    sentiments = df[text_col].apply(lambda x: analyzer.polarity_scores(str(x)) if pd.notna(x)
                                    else {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1})

    df['sentiment_score'] = sentiments.apply(lambda x: x['compound'])
    df['sentiment_pos'] = sentiments.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiments.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiments.apply(lambda x: x['neu'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )

    return df


def get_sentiment_summary(df):
    """
    Get summary statistics for sentiment analysis results.

    Args:
        df: DataFrame with sentiment_label and sentiment_score columns

    Returns:
        dict with summary statistics
    """
    if 'sentiment_label' not in df.columns:
        return {'error': 'No sentiment_label column found'}

    label_counts = df['sentiment_label'].value_counts().to_dict()

    return {
        'total_articles': len(df),
        'positive_count': label_counts.get('Positive', 0),
        'negative_count': label_counts.get('Negative', 0),
        'neutral_count': label_counts.get('Neutral', 0),
        'positive_pct': round(label_counts.get('Positive', 0) / len(df) * 100, 1),
        'negative_pct': round(label_counts.get('Negative', 0) / len(df) * 100, 1),
        'neutral_pct': round(label_counts.get('Neutral', 0) / len(df) * 100, 1),
        'avg_compound_score': round(df['sentiment_score'].mean(), 4),
        'std_compound_score': round(df['sentiment_score'].std(), 4),
        'most_positive': df.loc[df['sentiment_score'].idxmax()].to_dict() if len(df) > 0 else None,
        'most_negative': df.loc[df['sentiment_score'].idxmin()].to_dict() if len(df) > 0 else None,
    }


def plot_sentiment_distribution(df, save_path=None):
    """
    Plot sentiment label distribution.

    Args:
        df: DataFrame with sentiment_label column
        save_path: optional path to save the figure
    """
    if 'sentiment_label' not in df.columns:
        logger.warning("No sentiment_label column found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = {'Positive': '#00C853', 'Neutral': '#FFD700', 'Negative': '#FF5252'}
    counts = df['sentiment_label'].value_counts()
    bars = axes[0].bar(counts.index, counts.values,
                       color=[colors.get(x, '#888') for x in counts.index],
                       alpha=0.8, edgecolor='white')
    axes[0].set_title('Sentiment Distribution', fontsize=14)
    axes[0].set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     str(val), ha='center', va='bottom', fontweight='bold')

    # Score histogram
    axes[1].hist(df['sentiment_score'], bins=30, color='#00D4FF', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0.05, color='#00C853', linestyle='--', label='Positive threshold')
    axes[1].axvline(x=-0.05, color='#FF5252', linestyle='--', label='Negative threshold')
    axes[1].set_title('Sentiment Score Distribution', fontsize=14)
    axes[1].set_xlabel('Compound Score')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Demo
    sample_headlines = [
        "Apple reports record Q4 earnings, stock surges 5%",
        "Tesla faces production delays, shares drop significantly",
        "Federal Reserve keeps interest rates unchanged",
        "NVIDIA AI chip demand exceeds expectations",
        "Market volatility increases amid trade war concerns"
    ]

    print("Sentiment Analysis Demo")
    print("=" * 60)

    for headline in sample_headlines:
        result = analyze_sentiment(headline)
        print(f"\n{headline}")
        print(f"  Score: {result['compound_score']:.4f} | Label: {result['label']}")

    batch_results = analyze_batch(sample_headlines)
    print(f"\n\nBatch Results:\n{batch_results[['text', 'compound_score', 'label']]}")
