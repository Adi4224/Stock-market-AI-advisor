"""
Sentiment Analysis Pipeline.

Provides a clean interface for analyzing financial news sentiment
using the backend VADER analyzer.

Author: Stock Market AI Advisor Team
"""

import os
import sys
import logging
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def analyze_news_sentiment(text):
    """
    Analyze sentiment of a single news text.

    Args:
        text: news headline or article text

    Returns:
        dict with sentiment results
    """
    try:
        from src.backend.sentiment_analysis import analyze_sentiment
        result = analyze_sentiment(text)

        return {
            'text': text,
            'sentiment_label': result['label'],
            'sentiment_score': result['compound_score'],
            'confidence': abs(result['compound_score']),
            'details': {
                'positive': result['pos'],
                'negative': result['neg'],
                'neutral': result['neu']
            }
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {
            'text': text,
            'sentiment_label': 'Neutral',
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'details': {'positive': 0, 'negative': 0, 'neutral': 1},
            'error': str(e)
        }


def analyze_multiple(texts):
    """
    Analyze sentiment for multiple texts.

    Args:
        texts: list of strings

    Returns:
        list of sentiment result dicts
    """
    return [analyze_news_sentiment(text) for text in texts]


def get_ticker_sentiment(ticker):
    """
    Get overall sentiment for a specific stock ticker from news data.

    Args:
        ticker: stock ticker symbol (e.g., 'AAPL')

    Returns:
        dict with ticker sentiment summary
    """
    try:
        news_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'news_data.csv')

        if not os.path.exists(news_path):
            return _get_demo_ticker_sentiment(ticker)

        df = pd.read_csv(news_path)

        if 'ticker' not in df.columns:
            return _get_demo_ticker_sentiment(ticker)

        ticker_news = df[df['ticker'].str.upper() == ticker.upper()]

        if ticker_news.empty:
            return {
                'ticker': ticker,
                'overall_sentiment': 'Neutral',
                'avg_score': 0.0,
                'total_articles': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'recent_headlines': [],
                'message': f'No news data found for {ticker}'
            }

        # Analyze all headlines
        from src.backend.sentiment_analysis import analyze_sentiment

        results = []
        for _, row in ticker_news.iterrows():
            sentiment = analyze_sentiment(str(row.get('headline', '')))
            results.append({
                'headline': row.get('headline', ''),
                'date': row.get('date', ''),
                'source': row.get('source', ''),
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['compound_score']
            })

        results_df = pd.DataFrame(results)

        pos = len(results_df[results_df['sentiment_label'] == 'Positive'])
        neg = len(results_df[results_df['sentiment_label'] == 'Negative'])
        neu = len(results_df[results_df['sentiment_label'] == 'Neutral'])
        avg_score = results_df['sentiment_score'].mean()

        if avg_score >= 0.05:
            overall = 'Positive'
        elif avg_score <= -0.05:
            overall = 'Negative'
        else:
            overall = 'Neutral'

        recent = results_df.sort_values('date', ascending=False).head(10).to_dict('records')

        return {
            'ticker': ticker,
            'overall_sentiment': overall,
            'avg_score': round(avg_score, 4),
            'total_articles': len(results_df),
            'positive_count': pos,
            'negative_count': neg,
            'neutral_count': neu,
            'recent_headlines': recent
        }

    except Exception as e:
        logger.error(f"Ticker sentiment error: {e}")
        return _get_demo_ticker_sentiment(ticker)


def _get_demo_ticker_sentiment(ticker):
    """Generate demo sentiment data when news data is unavailable."""
    import numpy as np
    np.random.seed(hash(ticker) % 2**31)

    demo_headlines = [
        f"{ticker} reports strong quarterly earnings, beating estimates",
        f"Analysts upgrade {ticker} to 'Buy' rating",
        f"{ticker} faces regulatory challenges in key markets",
        f"Market outlook for {ticker} remains cautiously optimistic",
        f"{ticker} announces new product line expansion",
    ]

    results = []
    for h in demo_headlines:
        try:
            from src.backend.sentiment_analysis import analyze_sentiment
            s = analyze_sentiment(h)
            results.append({
                'headline': h,
                'sentiment_label': s['label'],
                'sentiment_score': s['compound_score']
            })
        except Exception:
            results.append({
                'headline': h,
                'sentiment_label': 'Neutral',
                'sentiment_score': 0.0
            })

    pos = sum(1 for r in results if r['sentiment_label'] == 'Positive')
    neg = sum(1 for r in results if r['sentiment_label'] == 'Negative')

    return {
        'ticker': ticker,
        'overall_sentiment': 'Positive' if pos > neg else ('Negative' if neg > pos else 'Neutral'),
        'avg_score': round(np.mean([r['sentiment_score'] for r in results]), 4),
        'total_articles': len(results),
        'positive_count': pos,
        'negative_count': neg,
        'neutral_count': len(results) - pos - neg,
        'recent_headlines': results,
        'is_demo': True
    }


def get_market_sentiment():
    """
    Get overall market sentiment from all available news data.

    Returns:
        dict with market-wide sentiment summary
    """
    try:
        news_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'news_data.csv')

        if not os.path.exists(news_path):
            return {'overall_sentiment': 'Neutral', 'avg_score': 0.0, 'is_demo': True}

        df = pd.read_csv(news_path)
        from src.backend.sentiment_analysis import analyze_news_dataframe, get_sentiment_summary

        analyzed = analyze_news_dataframe(df, text_col='headline')
        summary = get_sentiment_summary(analyzed)

        overall = 'Positive' if summary['avg_compound_score'] >= 0.05 \
            else ('Negative' if summary['avg_compound_score'] <= -0.05 else 'Neutral')

        summary['overall_sentiment'] = overall

        # Per-ticker breakdown
        if 'ticker' in analyzed.columns:
            ticker_sentiments = {}
            for ticker in analyzed['ticker'].unique():
                t_data = analyzed[analyzed['ticker'] == ticker]
                avg = t_data['sentiment_score'].mean()
                ticker_sentiments[ticker] = {
                    'avg_score': round(avg, 4),
                    'label': 'Positive' if avg >= 0.05 else ('Negative' if avg <= -0.05 else 'Neutral'),
                    'count': len(t_data)
                }
            summary['ticker_breakdown'] = ticker_sentiments

        return summary

    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        return {'overall_sentiment': 'Neutral', 'avg_score': 0.0, 'error': str(e)}
