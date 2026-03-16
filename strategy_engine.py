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

Dependencies   : pandas, numpy  ← NO external TA library required.
                 All indicators (EMA, MACD, RSI) are implemented natively
                 using pandas .ewm() so there are zero compiled extensions
                 and no Python-version restrictions.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── No third-party TA library needed ─────────────────────────────────────────
# EMA, MACD, and RSI are implemented below using pandas + numpy only.
# This eliminates numba/compiled dependencies — cloud-safe on all Python versions.
# Do NOT add pandas-ta, ta, or pandas-ta-openbb to requirements.txt.

# ── Import Layer 1 ────────────────────────────────────────────────────────────
try:
    from data_fetcher import DATA_DIR, DEFAULT_INTERVAL, load_csv, log as _fetcher_log
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DEFAULT_INTERVAL = "1d"
    load_csv = None
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
    ema_short:     int   = 50
    ema_long:      int   = 200
    macd_fast:     int   = 12
    macd_slow:     int   = 26
    macd_signal:   int   = 9
    rsi_period:    int   = 14
    rsi_buy_low:   float = 50.0
    rsi_buy_high:  float = 65.0
    rsi_sell_low:  float = 35.0
    rsi_sell_high: float = 50.0
    ema_touch_pct: float = 0.002   # 0.2% tolerance band


CONFIG = StrategyConfig()


# ── Signal constants ──────────────────────────────────────────────────────────

SIGNAL_BUY:     int = 1
SIGNAL_SELL:    int = -1
SIGNAL_NEUTRAL: int = 0


# ── Input validation ──────────────────────────────────────────────────────────

def _validate_input(df: pd.DataFrame, caller: str) -> None:
    """Raise ValueError if required columns are absent or DataFrame is empty."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{caller}] Missing columns: {missing}")
    if df.empty:
        raise ValueError(f"[{caller}] DataFrame is empty.")
    null_pct = df["Close"].isna().mean() * 100
    if null_pct > 10:
        log.warning("[%s] Close column has %.1f%% NaN values.", caller, null_pct)


# ── Indicator calculation — native pandas/numpy, no external TA library ───────

def calculate_ema(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append EMA_50 and EMA_200 columns using pandas ewm().

    pandas ewm(span=N, adjust=False) is mathematically identical to the
    standard EMA used by TradingView, Bloomberg, and J.P. Morgan research.
    min_periods=N ensures warm-up bars are NaN (same behaviour as pandas-ta).
    """
    try:
        _validate_input(df, "calculate_ema")
        result = df.copy()

        result[f"EMA_{config.ema_short}"] = (
            result["Close"]
            .ewm(span=config.ema_short, adjust=False, min_periods=config.ema_short)
            .mean()
        )
        result[f"EMA_{config.ema_long}"] = (
            result["Close"]
            .ewm(span=config.ema_long, adjust=False, min_periods=config.ema_long)
            .mean()
        )

        log.debug(
            "EMA_%d: %d warm-up NaNs | EMA_%d: %d warm-up NaNs",
            config.ema_short, result[f"EMA_{config.ema_short}"].isna().sum(),
            config.ema_long,  result[f"EMA_{config.ema_long}"].isna().sum(),
        )
        return result

    except Exception as exc:
        log.error("calculate_ema failed: %s", exc)
        raise


def calculate_macd(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append MACD_line, MACD_signal_line, and MACD_hist columns.

    Standard MACD formula:
        MACD line   = EMA(close, fast) − EMA(close, slow)
        Signal line = EMA(MACD line, signal_period)
        Histogram   = MACD line − Signal line
    """
    try:
        _validate_input(df, "calculate_macd")
        result = df.copy()

        ema_fast    = result["Close"].ewm(span=config.macd_fast,   adjust=False, min_periods=config.macd_fast).mean()
        ema_slow    = result["Close"].ewm(span=config.macd_slow,   adjust=False, min_periods=config.macd_slow).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.macd_signal, adjust=False, min_periods=config.macd_signal).mean()

        result["MACD_line"]        = macd_line
        result["MACD_signal_line"] = signal_line
        result["MACD_hist"]        = macd_line - signal_line

        log.debug("MACD columns appended: MACD_line, MACD_signal_line, MACD_hist")
        return result

    except Exception as exc:
        log.error("calculate_macd failed: %s", exc)
        raise


def calculate_rsi(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Append RSI_14 column using Wilder's smoothing method.

    Wilder's RSI formula:
        delta    = close.diff()
        avg_gain = EWM(alpha=1/period) of positive deltas
        avg_loss = EWM(alpha=1/period) of absolute negative deltas
        RS       = avg_gain / avg_loss
        RSI      = 100 − (100 / (1 + RS))

    alpha=1/period in ewm() is exactly Wilder's smoothing (RMA),
    matching the RSI values produced by TradingView and MetaTrader.
    """
    try:
        _validate_input(df, "calculate_rsi")
        result = df.copy()

        delta    = result["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / config.rsi_period, adjust=False, min_periods=config.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1.0 / config.rsi_period, adjust=False, min_periods=config.rsi_period).mean()
        rs       = avg_gain / avg_loss.replace(0, float("nan"))

        result[f"RSI_{config.rsi_period}"] = 100.0 - (100.0 / (1.0 + rs))

        log.debug("RSI_%d appended (%d NaN warm-up bars).",
                  config.rsi_period, result[f"RSI_{config.rsi_period}"].isna().sum())
        return result

    except Exception as exc:
        log.error("calculate_rsi failed: %s", exc)
        raise


# ── Signal detection helpers ──────────────────────────────────────────────────

def _is_bullish_reversal(row: pd.Series) -> bool:
    """SKILL.md trigger: Close > Open (green candle body)."""
    return bool(row["Close"] > row["Open"])


def _is_bearish_reversal(row: pd.Series) -> bool:
    """SKILL.md trigger: Close < Open (red candle body)."""
    return bool(row["Close"] < row["Open"])


def _touches_ema(price_low: float, price_high: float,
                 ema_val: float, tolerance_pct: float) -> bool:
    """
    True if candle [Low, High] intersects the EMA ± tolerance% band.
    Prevents requiring a pixel-perfect touch of the EMA value.
    """
    band_lo = ema_val * (1.0 - tolerance_pct)
    band_hi = ema_val * (1.0 + tolerance_pct)
    return price_low <= band_hi and price_high >= band_lo


# ── Core signal generation ────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """
    Apply AstroGoldHH-Institutional-Strategy signal logic (SKILL.md §Execution Rules).

    BUY  (+1): Price > EMA50 > EMA200 · MACD_hist > 0 · RSI ∈ [50,65]
               · candle touches EMA50 band · bullish reversal candle
    SELL (-1): Price < EMA50 < EMA200 · MACD_hist < 0 · RSI ∈ [35,50]
               · candle tests EMA50 as resistance · bearish reversal candle

    All five conditions must be satisfied simultaneously on the same bar.
    A bar cannot carry both BUY and SELL — BUY is evaluated first.
    """
    try:
        _validate_input(df, "generate_signals")

        required = [f"EMA_{config.ema_short}", f"EMA_{config.ema_long}",
                    "MACD_hist", f"RSI_{config.rsi_period}"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"generate_signals requires indicator columns: {missing}. "
                "Run calculate_ema / calculate_macd / calculate_rsi first."
            )

        result    = df.copy()
        result["Signal"] = SIGNAL_NEUTRAL
        ema_s_col = f"EMA_{config.ema_short}"
        ema_l_col = f"EMA_{config.ema_long}"
        rsi_col   = f"RSI_{config.rsi_period}"
        tol       = config.ema_touch_pct
        buys = sells = 0

        for i in range(len(result)):
            row       = result.iloc[i]
            close     = row["Close"]
            ema_short = row[ema_s_col]
            ema_long  = row[ema_l_col]
            macd_hist = row["MACD_hist"]
            rsi       = row[rsi_col]

            if any(pd.isna(v) for v in [ema_short, ema_long, macd_hist, rsi]):
                continue

            # BUY
            if (close > ema_short > ema_long
                    and macd_hist > 0
                    and config.rsi_buy_low <= rsi <= config.rsi_buy_high
                    and _touches_ema(row["Low"], row["High"], ema_short, tol)
                    and _is_bullish_reversal(row)):
                result.iat[i, result.columns.get_loc("Signal")] = SIGNAL_BUY
                buys += 1
                log.debug("BUY  @ %s  Close=%.2f  RSI=%.1f  MACDh=%.4f",
                          result.index[i], close, rsi, macd_hist)
                continue

            # SELL
            if (close < ema_short < ema_long
                    and macd_hist < 0
                    and config.rsi_sell_low <= rsi <= config.rsi_sell_high
                    and _touches_ema(row["Low"], row["High"], ema_short, tol)
                    and _is_bearish_reversal(row)):
                result.iat[i, result.columns.get_loc("Signal")] = SIGNAL_SELL
                sells += 1
                log.debug("SELL @ %s  Close=%.2f  RSI=%.1f  MACDh=%.4f",
                          result.index[i], close, rsi, macd_hist)

        log.info("Signal scan complete — BUY: %d  SELL: %d  (of %d bars)",
                 buys, sells, len(result))
        return result

    except Exception as exc:
        log.error("generate_signals failed: %s", exc)
        raise


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_strategy(df: pd.DataFrame, config: StrategyConfig = CONFIG) -> pd.DataFrame:
    """Full Layer-2 pipeline: raw OHLCV → EMA → MACD → RSI → Signals."""
    log.info("=" * 60)
    log.info("AstroGoldHH — Strategy Engine  (AstroGoldHH-Institutional-Strategy)")
    log.info("EMA %d/%d  |  MACD %d/%d/%d  |  RSI %d",
             config.ema_short, config.ema_long,
             config.macd_fast, config.macd_slow, config.macd_signal,
             config.rsi_period)
    log.info("Input rows: %d", len(df))
    log.info("=" * 60)

    try:
        df = calculate_ema(df,  config); log.info("EMA indicators   ✓")
        df = calculate_macd(df, config); log.info("MACD indicators  ✓")
        df = calculate_rsi(df,  config); log.info("RSI indicator    ✓")
        df = generate_signals(df, config); log.info("Signal scan      ✓")

        buys  = (df["Signal"] == SIGNAL_BUY).sum()
        sells = (df["Signal"] == SIGNAL_SELL).sum()
        log.info("-" * 60)
        log.info("Output rows: %d  |  BUY: %d  |  SELL: %d", len(df), buys, sells)
        log.info("-" * 60)
        return df

    except Exception as exc:
        log.error("run_strategy pipeline failed: %s", exc)
        raise


def get_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a filtered DataFrame of only BUY/SELL signal rows."""
    try:
        if "Signal" not in df.columns:
            raise ValueError("No 'Signal' column. Run run_strategy() first.")

        keep = [c for c in [
            "Close",
            f"EMA_{CONFIG.ema_short}", f"EMA_{CONFIG.ema_long}",
            "MACD_hist", f"RSI_{CONFIG.rsi_period}", "Signal",
        ] if c in df.columns]

        signals = df.loc[df["Signal"] != SIGNAL_NEUTRAL, keep].copy()
        signals["Signal_Label"] = signals["Signal"].map({SIGNAL_BUY: "BUY", SIGNAL_SELL: "SELL"})
        return signals

    except Exception as exc:
        log.error("get_signal_summary failed: %s", exc)
        raise


# ── Entry point (CLI only — not called by web_app.py) ────────────────────────

if __name__ == "__main__":
    try:
        from data_fetcher import load_csv, DATA_DIR, DEFAULT_INTERVAL
    except ImportError:
        log.error("data_fetcher.py not found. Run data_fetcher.py first.")
        sys.exit(1)

    TICKER = "GC=F"
    try:
        raw_df = load_csv(TICKER, interval=DEFAULT_INTERVAL, data_dir=DATA_DIR)
    except FileNotFoundError as exc:
        log.error("%s\nRun:  python data_fetcher.py  first.", exc)
        sys.exit(1)

    result_df = run_strategy(raw_df)
    summary   = get_signal_summary(result_df)

    print("\n── AstroGoldHH Signal Summary ─────────────────────────────")
    if summary.empty:
        print("  No signals generated. Try a longer period (e.g. '5y').")
    else:
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 120)
        print(summary.to_string())
    print("─────────────────────────────────────────────────────────────")

    out_path = DATA_DIR / "GC_F_strategy_output.csv"
    result_df.to_csv(out_path)
    log.info("Strategy output saved → %s", out_path)
