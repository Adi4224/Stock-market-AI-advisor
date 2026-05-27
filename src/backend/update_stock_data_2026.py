"""
update_stock_data_2026.py
=========================
Script to update historical Kaggle stock data with the latest market data
fetched via yfinance, producing per-ticker CSVs and a single combined dataset
that extends through 2026.

Workflow
--------
1. Read per-ticker CSVs from ``data/raw/kaggle_stock_data/``.
2. For each ticker, determine the last available date and fetch new rows from
   yfinance (last_date + 1 → today).  If no Kaggle file exists the full
   history is downloaded as a fallback (demo mode).
3. Merge old + new data, deduplicate by date, and sort chronologically.
4. Save updated per-ticker CSVs to ``data/updated/yfinance_updates/``.
5. Combine every ticker into a single
   ``data/processed/final_stock_dataset_2026.csv`` with an added **Ticker**
   column.

Usage
-----
    python src/backend/update_stock_data_2026.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None
    logging.warning(
        "yfinance is not installed. Install it with: pip install yfinance"
    )

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

# Project-root-relative directory paths
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_KAGGLE_DIR = os.path.join(_PROJECT_ROOT, "data", "raw", "kaggle_stock_data", "stocks")
_UPDATE_DIR = os.path.join(_PROJECT_ROOT, "data", "updated", "yfinance_updates")
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_COMBINED_CSV = os.path.join(_PROCESSED_DIR, "final_stock_dataset_2026.csv")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column naming and a proper Date column."""
    df.columns = [c.strip().title() for c in df.columns]

    # If the index looks like a DatetimeIndex, reset it into a column.
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Rename common yfinance variants
    rename_map = {
        "Adj Close": "Adj_Close",
        "Adj_close": "Adj_Close",
        "Adjclose": "Adj_Close",
    }
    df.rename(columns=rename_map, inplace=True)

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        try:
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
        except AttributeError:
            pass
        df["Date"] = dates
        df.dropna(subset=["Date"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def update_single_ticker(
    ticker: str,
    kaggle_dir: str = _KAGGLE_DIR,
    output_dir: str = _UPDATE_DIR,
) -> pd.DataFrame:
    """Fetch new data for *ticker* and merge it with existing Kaggle data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    kaggle_dir : str
        Directory containing per-ticker Kaggle CSVs.
    output_dir : str
        Directory where the updated per-ticker CSV will be written.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame (sorted by date, duplicates removed).
    """
    if yf is None:
        logger.error("yfinance is not available – cannot fetch data for %s.", ticker)
        return pd.DataFrame()

    _ensure_dirs(output_dir)

    kaggle_path = os.path.join(kaggle_dir, f"{ticker}.csv")
    old_df = pd.DataFrame()
    fetch_start = None  # None → full history

    # ------------------------------------------------------------------
    # 1. Try to load existing Kaggle data
    # ------------------------------------------------------------------
    if os.path.isfile(kaggle_path):
        try:
            old_df = pd.read_csv(kaggle_path)
            old_df = _normalise_columns(old_df)
            if "Date" in old_df.columns and not old_df.empty:
                last_date = old_df["Date"].max()
                fetch_start = last_date + timedelta(days=1)
                logger.info(
                    "[%s] Kaggle data found (%d rows, last date: %s). "
                    "Fetching from %s.",
                    ticker,
                    len(old_df),
                    last_date.strftime("%Y-%m-%d"),
                    fetch_start.strftime("%Y-%m-%d"),
                )
        except Exception as exc:
            logger.warning(
                "[%s] Failed to read Kaggle CSV (%s). Falling back to full "
                "yfinance download.",
                ticker,
                exc,
            )
            old_df = pd.DataFrame()
    else:
        logger.info(
            "[%s] No Kaggle CSV found – downloading full history (demo mode).",
            ticker,
        )

    # ------------------------------------------------------------------
    # 2. Fetch new data from yfinance
    # ------------------------------------------------------------------
    try:
        yf_ticker = yf.Ticker(ticker)
        if fetch_start is not None:
            today_str = datetime.today().strftime("%Y-%m-%d")
            start_str = fetch_start.strftime("%Y-%m-%d")
            if fetch_start > datetime.today():
                logger.info("[%s] Already up-to-date.", ticker)
                new_df = pd.DataFrame()
            else:
                new_df = yf_ticker.history(start=start_str, end=today_str)
        else:
            new_df = yf_ticker.history(period="max")

        if new_df is not None and not new_df.empty:
            new_df = _normalise_columns(new_df)
            logger.info("[%s] Fetched %d new rows from yfinance.", ticker, len(new_df))
        else:
            new_df = pd.DataFrame()
            logger.info("[%s] No new rows returned by yfinance.", ticker)

    except Exception as exc:
        logger.error("[%s] yfinance fetch failed: %s", ticker, exc)
        new_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Merge, deduplicate, sort
    # ------------------------------------------------------------------
    if old_df.empty and new_df.empty:
        logger.warning("[%s] No data available at all.", ticker)
        return pd.DataFrame()

    merged = pd.concat([old_df, new_df], ignore_index=True)
    if "Date" in merged.columns:
        merged.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        merged.sort_values("Date", inplace=True)
        merged.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 4. Save to output directory
    # ------------------------------------------------------------------
    out_path = os.path.join(output_dir, f"{ticker}.csv")
    merged.to_csv(out_path, index=False)
    logger.info("[%s] Saved %d rows → %s", ticker, len(merged), out_path)

    return merged


def update_all_tickers(
    kaggle_dir: str = _KAGGLE_DIR,
    output_dir: str = _UPDATE_DIR,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Update every ticker and return the combined DataFrame.

    Parameters
    ----------
    kaggle_dir : str
        Directory containing per-ticker Kaggle CSVs.
    output_dir : str
        Directory where updated per-ticker CSVs will be saved.
    tickers : list[str] or None
        Tickers to process.  Defaults to :pydata:`DEFAULT_TICKERS`.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a ``Ticker`` column.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    all_frames: list[pd.DataFrame] = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        logger.info(
            "── Processing %s (%d/%d) ──────────────────────────",
            ticker,
            idx,
            total,
        )
        try:
            df = update_single_ticker(ticker, kaggle_dir, output_dir)
            if not df.empty:
                df["Ticker"] = ticker
                all_frames.append(df)
        except Exception as exc:
            logger.error("[%s] Unexpected error: %s", ticker, exc)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        logger.info(
            "All tickers processed. Combined dataset: %d rows.", len(combined)
        )
        return combined

    logger.warning("No data collected for any ticker.")
    return pd.DataFrame()


def create_combined_dataset(
    update_dir: str = _UPDATE_DIR,
    output_path: str = _COMBINED_CSV,
) -> pd.DataFrame:
    """Read all per-ticker CSVs from *update_dir* and combine them.

    Parameters
    ----------
    update_dir : str
        Directory containing updated per-ticker CSVs.
    output_path : str
        Path for the combined CSV output.

    Returns
    -------
    pd.DataFrame
        The combined DataFrame.
    """
    _ensure_dirs(os.path.dirname(output_path))

    frames: list[pd.DataFrame] = []
    if not os.path.isdir(update_dir):
        logger.warning("Update directory does not exist: %s", update_dir)
        return pd.DataFrame()

    csv_files = sorted(
        f for f in os.listdir(update_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        logger.warning("No CSV files found in %s", update_dir)
        return pd.DataFrame()

    for fname in csv_files:
        ticker = os.path.splitext(fname)[0]
        fpath = os.path.join(update_dir, fname)
        try:
            df = pd.read_csv(fpath)
            df = _normalise_columns(df)
            if "Ticker" not in df.columns:
                df["Ticker"] = ticker
            frames.append(df)
            logger.info("Read %s – %d rows.", fname, len(df))
        except Exception as exc:
            logger.warning("Skipping %s: %s", fname, exc)

    if not frames:
        logger.warning("No valid data frames to combine.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "Date" in combined.columns:
        combined.sort_values(["Ticker", "Date"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

    combined.to_csv(output_path, index=False)
    logger.info(
        "Combined dataset saved (%d rows, %d tickers) → %s",
        len(combined),
        combined["Ticker"].nunique(),
        output_path,
    )
    return combined


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full update pipeline from the command line."""
    logger.info("=" * 60)
    logger.info("Stock Data Update Pipeline – started")
    logger.info("=" * 60)
    logger.info("Project root : %s", _PROJECT_ROOT)
    logger.info("Kaggle dir   : %s", _KAGGLE_DIR)
    logger.info("Output dir   : %s", _UPDATE_DIR)
    logger.info("Combined CSV : %s", _COMBINED_CSV)
    logger.info("Tickers      : %s", ", ".join(DEFAULT_TICKERS))
    logger.info("=" * 60)

    combined = update_all_tickers()

    if not combined.empty:
        create_combined_dataset()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
