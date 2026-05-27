"""
data_fetcher.py
===============
Utility module for fetching live and historical stock market data via the
`yfinance <https://pypi.org/project/yfinance/>`_ library.

This module exposes a small, well-documented API that the rest of the
Stock Market AI Advisor project can use to retrieve price data and company
information for one or more tickers.

Functions
---------
- :func:`fetch_live_stock_data`   – historical OHLCV for a single ticker.
- :func:`fetch_stock_info`        – company metadata / fundamentals.
- :func:`fetch_multiple_tickers`  – batch download for several tickers.
- :func:`get_default_tickers`     – canonical list of tracked tickers.

Example
-------
>>> from src.backend.data_fetcher import fetch_live_stock_data
>>> df = fetch_live_stock_data("AAPL", period="6mo")
>>> df.head()
"""

import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "CRM",
    "ADBE", "ORCL", "NFLX", "PYPL", "UBER",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP", "C", "BLK",
    # Healthcare
    "JNJ", "PFE", "UNH", "MRK", "ABT", "LLY", "ABBV", "TMO", "MDT", "BMY",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Consumer / Entertainment
    "WMT", "PG", "KO", "PEP", "COST", "HD", "NKE", "MCD", "SBUX", "DIS",
]

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_live_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    period : str, optional
        Data period to download.  Valid values include ``"1d"``, ``"5d"``,
        ``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``,
        ``"10y"``, ``"ytd"``, ``"max"``.  Defaults to ``"1y"``.
    interval : str, optional
        Data granularity.  Common values: ``"1d"``, ``"1wk"``, ``"1mo"``.
        Defaults to ``"1d"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns such as Open, High, Low, Close, Volume, and
        an index or column named Date.  Returns an empty DataFrame on failure.

    Raises
    ------
    RuntimeError
        If ``yfinance`` is not installed.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Install it with: pip install yfinance"
        )

    logger.info("Fetching %s data for %s (interval=%s)…", period, ticker, interval)

    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(period=period, interval=interval)

        if df is None or df.empty:
            logger.warning("[%s] No data returned by yfinance.", ticker)
            return pd.DataFrame()

        # Reset DatetimeIndex into a column for downstream consistency.
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        df.columns = [c.strip().title() for c in df.columns]
        df['Ticker'] = ticker
        logger.info("[%s] Retrieved %d rows.", ticker, len(df))
        return df

    except Exception as exc:
        logger.error("[%s] Failed to fetch data: %s", ticker, exc)
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> dict:
    """Retrieve company metadata and key financial metrics.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        A dictionary of company information fields (e.g. ``longName``,
        ``sector``, ``marketCap``).  Returns an empty dict on failure.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Install it with: pip install yfinance"
        )

    logger.info("Fetching info for %s…", ticker)

    try:
        yf_ticker = yf.Ticker(ticker)
        info: dict = yf_ticker.info or {}
        logger.info(
            "[%s] Info retrieved – %d fields.", ticker, len(info)
        )
        return info

    except Exception as exc:
        logger.error("[%s] Failed to fetch info: %s", ticker, exc)
        return {}


def fetch_multiple_tickers(
    tickers: Optional[list[str]] = None,
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download data for several tickers and return a mapping.

    Parameters
    ----------
    tickers : list[str] or None
        Tickers to download.  Defaults to :pydata:`DEFAULT_TICKERS`.
    period : str, optional
        See :func:`fetch_live_stock_data`.
    interval : str, optional
        See :func:`fetch_live_stock_data`.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{ticker: DataFrame}`` for every ticker that returned data.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    results: dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        logger.info("── %s (%d/%d) ──", ticker, idx, total)
        try:
            df = fetch_live_stock_data(ticker, period=period, interval=interval)
            if not df.empty:
                results[ticker] = df
        except Exception as exc:
            logger.error("[%s] Skipped due to error: %s", ticker, exc)

    logger.info(
        "Batch download complete: %d/%d tickers returned data.",
        len(results),
        total,
    )
    return results


def get_default_tickers() -> list[str]:
    """Return the canonical list of tracked stock tickers.

    Returns
    -------
    list[str]
        A copy of :pydata:`DEFAULT_TICKERS`.
    """
    return list(DEFAULT_TICKERS)


# ---------------------------------------------------------------------------
# Convenience CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Default tickers:", get_default_tickers())
    print()

    # Quick demo – fetch 1 month of AAPL data
    demo_ticker = "AAPL"
    df = fetch_live_stock_data(demo_ticker, period="1mo")
    if not df.empty:
        print(f"\n{demo_ticker} – last 5 rows:\n")
        print(df.tail().to_string(index=False))
    else:
        print(f"No data returned for {demo_ticker}.")
