"""
News Data Preprocessing Module
===============================
Provides functions for cleaning, tokenizing, and preprocessing financial news
headlines. Uses NLTK for natural language processing tasks including
tokenization and stopword removal.
"""

import re
import pandas as pd
import nltk

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Cache English stopwords for performance
_ENGLISH_STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """
    Clean a single text string for downstream NLP tasks.

    Performs the following transformations:
        1. Converts text to lowercase.
        2. Removes URLs (http/https links).
        3. Removes special characters and digits (keeps only letters and spaces).
        4. Collapses multiple whitespace characters into a single space.
        5. Strips leading and trailing whitespace.

    Args:
        text: The raw text string to clean.

    Returns:
        The cleaned text string. Returns an empty string if input is None
        or not a string.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str) -> list:
    """
    Tokenize a text string into individual words.

    Uses NLTK's word_tokenize for robust tokenization that handles
    contractions and punctuation correctly.

    Args:
        text: The text string to tokenize (should be pre-cleaned).

    Returns:
        A list of word tokens. Returns an empty list if input is None
        or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        return word_tokenize(text)
    except Exception:
        # Fallback to simple whitespace splitting
        return text.split()


def remove_stopwords(tokens: list) -> list:
    """
    Remove English stopwords from a list of tokens.

    Args:
        tokens: A list of word tokens.

    Returns:
        A filtered list with stopwords removed. Returns an empty list
        if input is None or not a list.
    """
    if not isinstance(tokens, list):
        return []

    return [token for token in tokens if token not in _ENGLISH_STOPWORDS]


def preprocess_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame of financial news data.

    Performs the following steps:
        1. Validates that the 'headline' column exists.
        2. Handles missing or empty headlines by replacing them with empty strings.
        3. Cleans each headline and stores the result in a 'cleaned_headline' column.
        4. Tokenizes cleaned headlines, removes stopwords, and stores the result
           in a 'tokens' column.

    Args:
        df: DataFrame containing at least a 'headline' column with news text.

    Returns:
        The DataFrame with added 'cleaned_headline' and 'tokens' columns.

    Raises:
        ValueError: If the 'headline' column is missing from the DataFrame.
    """
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain a 'headline' column.")

    df = df.copy()

    # Handle missing / empty headlines
    df['headline'] = df['headline'].fillna('').astype(str)

    # Clean headlines
    df['cleaned_headline'] = df['headline'].apply(clean_text)

    # Tokenize and remove stopwords
    df['tokens'] = (
        df['cleaned_headline']
        .apply(tokenize_text)
        .apply(remove_stopwords)
    )

    return df
