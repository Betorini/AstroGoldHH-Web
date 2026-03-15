"""
AstroGoldHH — Layer 1: Data Fetcher
=====================================
Responsibility : Fetch OHLCV market data from Yahoo Finance and persist it
                 to the data/ directory as CSV files.
Architecture   : L4 Agents — this module is stateless and has no strategy logic.
                 It only fetches, validates, and saves raw market data.

Dependencies   : yfinance, pandas
Usage          : python data_fetcher.py
                 or import and call fetch_ticker() / fetch_all() directly.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Optional dependency guard ────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError as exc:
    print("[FATAL] yfinance is not installed.  Run:  pip install yfinance pandas")
    sys.exit(1)


# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"

GOLD_TICKER: str = "GC=F"          # Gold Futures (USD / troy oz)
DEFAULT_PERIOD: str = "2y"         # lookback window
DEFAULT_INTERVAL: str = "1d"       # bar size

# Additional equity / ETF tickers that feed the strategy
EQUITY_TICKERS: List[str] = [
    "GLD",   # SPDR Gold Shares ETF — proxy for spot gold
    "SPY",   # S&P 500 ETF — macro risk-on/risk-off context
    "DXY",   # USD Index via ^DXY (correlation with gold)
    "TLT",   # 20yr Treasury ETF — real-rate proxy
]

REQUIRED_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logger(name: str = "data_fetcher") -> logging.Logger:
    """Return a logger that writes to both console and a rotating log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


log: logging.Logger = _configure_logger()


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FetchConfig:
    """Immutable configuration for a single fetch job."""
    ticker: str
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL
    output_dir: Path = DATA_DIR
    auto_adjust: bool = True          # adjust OHLC for splits & dividends

    def csv_path(self) -> Path:
        """Derive the output CSV path from ticker and interval."""
        safe_ticker = self.ticker.replace("=", "_").replace("^", "")
        return self.output_dir / f"{safe_ticker}_{self.interval}.csv"


@dataclass
class FetchResult:
    """Outcome returned from a single fetch operation."""
    ticker: str
    success: bool
    rows: int = 0
    csv_path: Optional[Path] = None
    error: Optional[str] = None


# ── Core fetch logic ──────────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Validate and normalise a raw yfinance DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or the DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError(f"Empty DataFrame returned for '{ticker}'.")

    # yfinance may return multi-level columns when downloading a single ticker;
    # flatten them if necessary.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns for '{ticker}': {missing}. "
            f"Got: {list(df.columns)}"
        )

    # Drop rows where ALL price columns are NaN
    df = df.dropna(subset=list(REQUIRED_COLUMNS), how="all")

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "Date"
    df.sort_index(inplace=True)
    return df


def fetch_ticker(config: FetchConfig) -> FetchResult:
    """
    Download OHLCV data for a single ticker and save it to CSV.

    Parameters
    ----------
    config : FetchConfig
        Fetch parameters (ticker, period, interval, output directory).

    Returns
    -------
    FetchResult
        Structured outcome containing success flag, row count, and path.
    """
    log.info("Fetching %-12s | period=%-5s | interval=%s",
             config.ticker, config.period, config.interval)

    try:
        raw: pd.DataFrame = yf.download(
            config.ticker,
            period=config.period,
            interval=config.interval,
            auto_adjust=config.auto_adjust,
            progress=False,
            threads=False,
        )

        df = _validate_dataframe(raw, config.ticker)

    except Exception as exc:
        log.error("Download failed for '%s': %s", config.ticker, exc)
        return FetchResult(ticker=config.ticker, success=False, error=str(exc))

    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = config.csv_path()
        df.to_csv(csv_path)
        log.info("Saved  %-12s → %s  (%d rows)", config.ticker, csv_path.name, len(df))
        return FetchResult(
            ticker=config.ticker,
            success=True,
            rows=len(df),
            csv_path=csv_path,
        )

    except OSError as exc:
        log.error("Failed to write CSV for '%s': %s", config.ticker, exc)
        return FetchResult(ticker=config.ticker, success=False, error=str(exc))


def fetch_all(
    tickers: Optional[List[str]] = None,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    output_dir: Path = DATA_DIR,
) -> Dict[str, FetchResult]:
    """
    Fetch data for multiple tickers and return a results map.

    Parameters
    ----------
    tickers    : list of Yahoo Finance ticker symbols; defaults to
                 GOLD_TICKER + EQUITY_TICKERS.
    period     : yfinance period string (e.g. "1y", "2y", "6mo").
    interval   : bar size string (e.g. "1d", "1h", "5m").
    output_dir : destination directory for CSV files.

    Returns
    -------
    dict[ticker -> FetchResult]
    """
    if tickers is None:
        tickers = [GOLD_TICKER] + EQUITY_TICKERS

    log.info("=" * 60)
    log.info("AstroGoldHH — Data Fetcher  [%s]", datetime.now().isoformat(timespec="seconds"))
    log.info("Tickers   : %s", tickers)
    log.info("Period    : %s  |  Interval : %s", period, interval)
    log.info("Output    : %s", output_dir)
    log.info("=" * 60)

    results: Dict[str, FetchResult] = {}
    for ticker in tickers:
        cfg = FetchConfig(
            ticker=ticker,
            period=period,
            interval=interval,
            output_dir=output_dir,
        )
        results[ticker] = fetch_ticker(cfg)

    # ── Summary report ────────────────────────────────────────────────────────
    ok = [r for r in results.values() if r.success]
    fail = [r for r in results.values() if not r.success]

    log.info("-" * 60)
    log.info("Completed: %d/%d succeeded", len(ok), len(tickers))
    for r in fail:
        log.warning("  FAILED  %-12s  %s", r.ticker, r.error)
    log.info("-" * 60)

    return results


# ── Convenience loader (used by strategy_engine.py) ───────────────────────────

def load_csv(ticker: str, interval: str = DEFAULT_INTERVAL,
             data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load a previously saved CSV for *ticker* and return it as a DataFrame.

    Parameters
    ----------
    ticker   : Yahoo Finance ticker symbol (e.g. "GC=F").
    interval : bar interval used when the file was saved.
    data_dir : directory to search in.

    Returns
    -------
    pd.DataFrame with DatetimeIndex, or raises FileNotFoundError.
    """
    safe_ticker = ticker.replace("=", "_").replace("^", "")
    csv_path = data_dir / f"{safe_ticker}_{interval}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No cached data found for '{ticker}'.  "
            f"Run data_fetcher.fetch_all() first.\n  Expected: {csv_path}"
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    log.debug("Loaded %s  (%d rows) from %s", ticker, len(df), csv_path.name)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = fetch_all()

    print("\n── Fetch Summary ──────────────────────────────────────")
    for ticker, r in results.items():
        status = "✓" if r.success else "✗"
        detail = (f"{r.rows} rows  →  {r.csv_path.name}"
                  if r.success else f"ERROR: {r.error}")
        print(f"  {status}  {ticker:<12}  {detail}")
    print("───────────────────────────────────────────────────────")
