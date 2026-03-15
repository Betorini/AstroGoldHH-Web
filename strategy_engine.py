"""
AstroGoldHH — Layer 2: Strategy Engine
=========================================
Responsibility : Calculate technical indicators (EMA 50/200, MACD 12/26/9,
                 RSI 14) and generate Buy/Sell signals per the
                 AstroGoldHH-Institutional-Strategy SKILL.md.

Architecture   : L4 Agents — this module consumes data from Layer 1
                 (data_fetcher) and produces a signal-annotated DataFrame
                 for Layer 3 (visualizer).  No I/O or chart logic here.

Strategy ref   : .claude/skills/SKILL.md  (AstroGoldHH-Institutional-Strategy)
                 J.P. Morgan institutional "Trend-Following & Mean-Reversion"

Dependencies   : pandas, numpy, pandas_ta
                 Install: pip install pandas numpy pandas-ta
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional dependency guard ────────────────────────────────────────────────
try:
    import pandas_ta as ta  # type: ignore[import]
except ImportError:
    print("[FATAL] pandas_ta is not installed.  Run:  pip install pandas-ta")
    sys.exit(1)

# ── Import Layer 1 ────────────────────────────────────────────────────────────
try:
    from data_fetcher import DATA_DIR, DEFAULT_INTERVAL, load_csv, log as _fetcher_log
except ImportError:
    # Allow running strategy_engine standalone for testing
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DEFAULT_INTERVAL = "1d"
    load_csv = None  # handled below
    _fetcher_log = None


# ── Logger ─────────────────────────────────────────────────────────────────────

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("strategy_engine")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        fh = logging.FileHandler(
            log_dir / f"strategy_engine_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log: logging.Logger = _get_logger()


# ── Strategy configuration (from SKILL.md) ───────────────────────────────────

@dataclass(frozen=True)
class StrategyConfig:
    """
    Immutable parameters sourced directly from AstroGoldHH SKILL.md.
    Change values here only when the SKILL.md is updated.
    """
    # Trend filters
    ema_short: int = 50
    ema_long: int = 200

    # Momentum — MACD (fast, slow, signal)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Oscillator
    rsi_period: int = 14

    # RSI gate bands (SKILL.md §Execution Rules)
    rsi_buy_low: float = 50.0
    rsi_buy_high: float = 65.0
    rsi_sell_low: float = 35.0
    rsi_sell_high: float = 50.0

    # EMA touch tolerance (% of price) — price considered "at" EMA if within band
    ema_touch_pct: float = 0.002   # 0.2 %


CONFIG = StrategyConfig()


# ── Signal constants ──────────────────────────────────────────────────────────

SIGNAL_BUY: int = 1
SIGNAL_SELL: int = -1
SIGNAL_NEUTRAL: int = 0


# ── Indicator calculation functions ──────────────────────────────────────────

def _validate_input(df: pd.DataFrame, caller: str) -> None:
    """
    Raise ValueError if required price columns are absent or DataFrame is empty.
    Called at the top of every public function (CLAUDE.md Data Integrity rule).
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{caller}] Missing columns: {missing}")
    if df.empty:
        raise ValueError(f"[{caller}] DataFrame is empty.")
    null_pct = df["Close"].isna().mean() * 100
    if null_pct > 10:
        log.warning("[%s] Close column has %.1f%% NaN values.", caller, null_pct)


def calculate_ema(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append EMA_50 and EMA_200 columns to *df*.

    Parameters
    ----------
    df     : OHLCV DataFrame (DatetimeIndex).
    config : StrategyConfig (uses ema_short / ema_long).

    Returns
    -------
    DataFrame with two additional columns: EMA_50, EMA_200.
    """
    try:
        _validate_input(df, "calculate_ema")
        result = df.copy()

        result[f"EMA_{config.ema_short}"] = (
            ta.ema(result["Close"], length=config.ema_short)
        )
        result[f"EMA_{config.ema_long}"] = (
            ta.ema(result["Close"], length=config.ema_long)
        )

        nan_short = result[f"EMA_{config.ema_short}"].isna().sum()
        nan_long  = result[f"EMA_{config.ema_long}"].isna().sum()
        log.debug("EMA_%d: %d warm-up NaNs | EMA_%d: %d warm-up NaNs",
                  config.ema_short, nan_short, config.ema_long, nan_long)
        return result

    except Exception as exc:
        log.error("calculate_ema failed: %s", exc)
        raise


def calculate_macd(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append MACD_line, MACD_signal, and MACD_hist columns to *df*.

    Uses pandas_ta which returns columns named:
        MACD_{fast}_{slow}_{signal}
        MACDs_{fast}_{slow}_{signal}
        MACDh_{fast}_{slow}_{signal}

    We rename to generic MACD_line / MACD_signal / MACD_hist for portability.
    """
    try:
        _validate_input(df, "calculate_macd")
        result = df.copy()

        macd_df: pd.DataFrame = ta.macd(
            result["Close"],
            fast=config.macd_fast,
            slow=config.macd_slow,
            signal=config.macd_signal,
        )

        if macd_df is None or macd_df.empty:
            raise ValueError("pandas_ta.macd() returned empty result.")

        # Rename columns defensively regardless of pandas_ta version
        col_map = {}
        for col in macd_df.columns:
            lower = col.lower()
            if lower.startswith("macdh"):
                col_map[col] = "MACD_hist"
            elif lower.startswith("macds"):
                col_map[col] = "MACD_signal_line"
            elif lower.startswith("macd"):
                col_map[col] = "MACD_line"
        macd_df = macd_df.rename(columns=col_map)

        result = pd.concat([result, macd_df], axis=1)
        log.debug("MACD columns appended: %s", list(macd_df.columns))
        return result

    except Exception as exc:
        log.error("calculate_macd failed: %s", exc)
        raise


def calculate_rsi(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append RSI_14 column to *df*.
    """
    try:
        _validate_input(df, "calculate_rsi")
        result = df.copy()

        rsi_series: pd.Series = ta.rsi(result["Close"], length=config.rsi_period)
        if rsi_series is None or rsi_series.empty:
            raise ValueError("pandas_ta.rsi() returned empty result.")

        result[f"RSI_{config.rsi_period}"] = rsi_series
        log.debug("RSI_%d appended (%d NaNs warm-up).",
                  config.rsi_period, rsi_series.isna().sum())
        return result

    except Exception as exc:
        log.error("calculate_rsi failed: %s", exc)
        raise


# ── Signal detection helpers ──────────────────────────────────────────────────

def _is_bullish_reversal(row: pd.Series) -> bool:
    """
    SKILL.md trigger — bullish reversal candle:
    Close > Open (green body).
    """
    return bool(row["Close"] > row["Open"])


def _is_bearish_reversal(row: pd.Series) -> bool:
    """
    SKILL.md trigger — bearish reversal candle:
    Close < Open (red body).
    """
    return bool(row["Close"] < row["Open"])


def _touches_ema(price_low: float, price_high: float,
                 ema_val: float, tolerance_pct: float) -> bool:
    """
    Return True if the candle's [Low, High] range touches or overlaps the EMA band.

    The EMA band = [ema_val * (1 - tol), ema_val * (1 + tol)].
    A touch occurs when the candle's range intersects this band.
    """
    band_lo = ema_val * (1.0 - tolerance_pct)
    band_hi = ema_val * (1.0 + tolerance_pct)
    return price_low <= band_hi and price_high >= band_lo


# ── Core signal generation ────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Apply the full AstroGoldHH-Institutional-Strategy signal logic and
    return the DataFrame with a new integer column ``Signal``:

        +1  = BUY  (Institutional Long)
        -1  = SELL (Institutional Short)
         0  = No signal

    Signal logic per SKILL.md §Execution Rules:

    BUY  — ALL conditions must hold on the same bar:
        1. Price > EMA_50 > EMA_200         (bullish trend alignment)
        2. MACD_hist > 0                     (expanding positive momentum)
        3. RSI in [50, 65]                   (momentum confirmed, not extended)
        4. Candle Low touches EMA_50 band    (pullback trigger)
        5. Close > Open                      (bullish reversal candle)

    SELL — ALL conditions must hold on the same bar:
        1. Price < EMA_50 < EMA_200          (bearish trend alignment)
        2. MACD_hist < 0                     (expanding negative momentum)
        3. RSI in [35, 50]                   (momentum confirmed, not oversold)
        4. Candle High touches EMA_50 band   (rally-to-resistance trigger)
        5. Close < Open                      (bearish reversal candle)

    Parameters
    ----------
    df     : DataFrame already containing all indicator columns.
             Must have been processed by calculate_ema / calculate_macd /
             calculate_rsi before being passed here.
    config : StrategyConfig (signal gate parameters).

    Returns
    -------
    DataFrame with Signal column added.
    """
    try:
        _validate_input(df, "generate_signals")

        required_indicators = [
            f"EMA_{config.ema_short}", f"EMA_{config.ema_long}",
            "MACD_hist", f"RSI_{config.rsi_period}",
        ]
        missing = [c for c in required_indicators if c not in df.columns]
        if missing:
            raise ValueError(
                f"generate_signals requires indicator columns: {missing}. "
                "Run calculate_ema / calculate_macd / calculate_rsi first."
            )

        result = df.copy()
        result["Signal"] = SIGNAL_NEUTRAL

        ema_s_col  = f"EMA_{config.ema_short}"
        ema_l_col  = f"EMA_{config.ema_long}"
        rsi_col    = f"RSI_{config.rsi_period}"
        tol        = config.ema_touch_pct

        buy_count = sell_count = 0

        for i in range(len(result)):
            row       = result.iloc[i]
            close     = row["Close"]
            open_     = row["Open"]
            low       = row["Low"]
            high      = row["High"]
            ema_short = row[ema_s_col]
            ema_long  = row[ema_l_col]
            macd_hist = row["MACD_hist"]
            rsi       = row[rsi_col]

            # Skip bars with NaN indicators (warm-up period)
            if any(pd.isna(v) for v in [ema_short, ema_long, macd_hist, rsi]):
                continue

            # ── BUY signal ────────────────────────────────────────────────
            buy_trend     = close > ema_short > ema_long
            buy_momentum  = macd_hist > 0
            buy_rsi       = config.rsi_buy_low <= rsi <= config.rsi_buy_high
            buy_touch     = _touches_ema(low, high, ema_short, tol)
            buy_candle    = _is_bullish_reversal(row)

            if buy_trend and buy_momentum and buy_rsi and buy_touch and buy_candle:
                result.iat[i, result.columns.get_loc("Signal")] = SIGNAL_BUY
                buy_count += 1
                log.debug("BUY  signal @ %s  Close=%.2f  RSI=%.1f  MACDh=%.4f",
                          result.index[i], close, rsi, macd_hist)
                continue  # a bar cannot be both BUY and SELL

            # ── SELL signal ───────────────────────────────────────────────
            sell_trend    = close < ema_short < ema_long
            sell_momentum = macd_hist < 0
            sell_rsi      = config.rsi_sell_low <= rsi <= config.rsi_sell_high
            sell_touch    = _touches_ema(low, high, ema_short, tol)
            sell_candle   = _is_bearish_reversal(row)

            if sell_trend and sell_momentum and sell_rsi and sell_touch and sell_candle:
                result.iat[i, result.columns.get_loc("Signal")] = SIGNAL_SELL
                sell_count += 1
                log.debug("SELL signal @ %s  Close=%.2f  RSI=%.1f  MACDh=%.4f",
                          result.index[i], close, rsi, macd_hist)

        log.info("Signal scan complete — BUY: %d  SELL: %d  (of %d bars)",
                 buy_count, sell_count, len(result))
        return result

    except Exception as exc:
        log.error("generate_signals failed: %s", exc)
        raise


# ── Pipeline: run all indicators + signals in one call ───────────────────────

def run_strategy(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Full Layer-2 pipeline:
        raw OHLCV  →  EMA  →  MACD  →  RSI  →  Signals

    Parameters
    ----------
    df     : Raw OHLCV DataFrame from Layer 1 (data_fetcher.load_csv).
    config : StrategyConfig overrides (optional).

    Returns
    -------
    DataFrame with all indicator and Signal columns appended.
    Suitable for direct consumption by Layer 3 (visualizer).
    """
    log.info("=" * 60)
    log.info("AstroGoldHH — Strategy Engine  (AstroGoldHH-Institutional-Strategy)")
    log.info("EMA %d/%d  |  MACD %d/%d/%d  |  RSI %d",
             config.ema_short, config.ema_long,
             config.macd_fast, config.macd_slow, config.macd_signal,
             config.rsi_period)
    log.info("Input rows: %d", len(df))
    log.info("=" * 60)

    try:
        df = calculate_ema(df, config)
        log.info("EMA indicators   ✓")

        df = calculate_macd(df, config)
        log.info("MACD indicators  ✓")

        df = calculate_rsi(df, config)
        log.info("RSI indicator    ✓")

        df = generate_signals(df, config)
        log.info("Signal scan      ✓")

        # ── Signal summary ────────────────────────────────────────────────
        buys  = (df["Signal"] == SIGNAL_BUY).sum()
        sells = (df["Signal"] == SIGNAL_SELL).sum()
        log.info("-" * 60)
        log.info("Output rows: %d  |  BUY signals: %d  |  SELL signals: %d",
                 len(df), buys, sells)
        log.info("-" * 60)

        return df

    except Exception as exc:
        log.error("run_strategy pipeline failed: %s", exc)
        raise


# ── Signal summary helper (for external consumers / tests) ────────────────────

def get_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a filtered DataFrame containing only signal rows (BUY / SELL).
    Useful for reporting and the visualizer's annotation layer.

    Columns returned: Date (index), Close, EMA_50, EMA_200,
                      MACD_hist, RSI_14, Signal
    """
    try:
        if "Signal" not in df.columns:
            raise ValueError("DataFrame has no 'Signal' column. Run run_strategy() first.")

        keep_cols = [
            "Close",
            f"EMA_{CONFIG.ema_short}",
            f"EMA_{CONFIG.ema_long}",
            "MACD_hist",
            f"RSI_{CONFIG.rsi_period}",
            "Signal",
        ]
        available = [c for c in keep_cols if c in df.columns]
        signals   = df.loc[df["Signal"] != SIGNAL_NEUTRAL, available].copy()

        signals["Signal_Label"] = signals["Signal"].map(
            {SIGNAL_BUY: "BUY", SIGNAL_SELL: "SELL"}
        )
        return signals

    except Exception as exc:
        log.error("get_signal_summary failed: %s", exc)
        raise


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Attempt to load data saved by data_fetcher.py (Layer 1)
    try:
        from data_fetcher import load_csv, DATA_DIR, DEFAULT_INTERVAL
    except ImportError:
        log.error("data_fetcher.py not found. Run data_fetcher.py first.")
        sys.exit(1)

    TICKER = "GC=F"
    log.info("Loading cached data for %s …", TICKER)

    try:
        raw_df = load_csv(TICKER, interval=DEFAULT_INTERVAL, data_dir=DATA_DIR)
    except FileNotFoundError as exc:
        log.error("%s\nRun:  python data_fetcher.py  first.", exc)
        sys.exit(1)

    # Run full strategy pipeline
    result_df = run_strategy(raw_df)

    # Print signal summary
    summary = get_signal_summary(result_df)
    print("\n── AstroGoldHH Signal Summary ─────────────────────────────")
    if summary.empty:
        print("  No signals generated for the selected period.")
        print("  Tip: Try a longer period (e.g. '5y') in data_fetcher.py.")
    else:
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 120)
        print(summary.to_string())
    print("─────────────────────────────────────────────────────────────")

    # Persist enriched DataFrame for Layer 3
    out_path = DATA_DIR / "GC_F_strategy_output.csv"
    result_df.to_csv(out_path)
    log.info("Strategy output saved → %s", out_path)
