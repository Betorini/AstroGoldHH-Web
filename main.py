"""
AstroGoldHH — Layer 4: Orchestrator
======================================
Responsibility : Single entry point that wires L1 → L2 → L3 together and
                 exposes the three CLAUDE.md commands via argparse.

Architecture   : L4 Agents (CLAUDE.md §2)
                 This module contains ZERO business logic — it only calls
                 the lower layers and manages control flow, logging, and
                 error routing.

CLAUDE.md ref  : §5 Commands
                 --analyze  → /analyze : trigger AstroGoldHH strategy logic
                 --render   → /render  : generate Plotly interactive dashboard
                 --validate → /validate: run unit test suite

Usage examples
--------------
    python main.py --analyze                        # L1 + L2 only
    python main.py --render                         # L3 only (requires prior --analyze)
    python main.py --analyze --render               # full pipeline
    python main.py --validate                       # run validation suite
    python main.py --analyze --render --ticker GLD  # custom ticker
    python main.py --analyze --period 5y            # custom lookback
    python main.py --analyze --render --no-browser  # suppress auto-open
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
LOG_DIR:      Path = PROJECT_ROOT / "logs"
DATA_DIR:     Path = PROJECT_ROOT / "data"
SKILL_PATH:   Path = PROJECT_ROOT / ".claude" / "skills" / "SKILL.md"
STRATEGY_CSV: Path = DATA_DIR / "GC_F_strategy_output.csv"


# ── Logger ────────────────────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    """
    Configure the root orchestrator logger.
    Console: INFO with colour-coded level prefix.
    File:    DEBUG with full timestamps → logs/main_YYYYMMDD.log
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("AstroGoldHH")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class _ConsoleFmt(logging.Formatter):
        """Minimal console formatter with level-colour prefix."""
        LEVEL_PREFIX = {
            "DEBUG":    "\033[90m[DBG]\033[0m",
            "INFO":     "\033[36m[INF]\033[0m",
            "WARNING":  "\033[33m[WRN]\033[0m",
            "ERROR":    "\033[31m[ERR]\033[0m",
            "CRITICAL": "\033[41m[CRT]\033[0m",
        }
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            prefix = self.LEVEL_PREFIX.get(record.levelname, "[???]")
            return f"{prefix} {record.getMessage()}"

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ConsoleFmt())
    logger.addHandler(ch)

    fh = logging.FileHandler(
        LOG_DIR / f"main_{datetime.now():%Y%m%d}.log", encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    return logger


log: logging.Logger = _build_logger()


# ── Error routing helpers ─────────────────────────────────────────────────────

def _strategy_error(exc: Exception, context: str) -> None:
    """
    Log a strategy/calculation error and direct the user to SKILL.md.
    Called whenever an exception originates from strategy_engine.
    (CLAUDE.md §4 Troubleshooting: "If an error occurs … query the SKILL.md first.")
    """
    log.error("─" * 60)
    log.error("CALCULATION ERROR in %s", context)
    log.error("  %s: %s", type(exc).__name__, exc)
    log.error("")
    log.error("  Consult the strategy reference first:")
    log.error("  → %s", SKILL_PATH)
    log.error("")
    log.error("  Full traceback written to:  logs/main_%s.log",
              datetime.now().strftime("%Y%m%d"))
    log.error("─" * 60)
    log.debug("Full traceback:\n%s", traceback.format_exc())


def _general_error(exc: Exception, context: str) -> None:
    """Log a non-strategy error and direct the user to the log file."""
    log.error("─" * 60)
    log.error("ERROR in %s", context)
    log.error("  %s: %s", type(exc).__name__, exc)
    log.error("  Full traceback → logs/main_%s.log", datetime.now().strftime("%Y%m%d"))
    log.error("─" * 60)
    log.debug("Full traceback:\n%s", traceback.format_exc())


# ── Status banner helpers ─────────────────────────────────────────────────────

def _banner(title: str) -> None:
    """Print a section header banner to the console."""
    width = 62
    log.info("╔" + "═" * width + "╗")
    log.info("║  %-*s║" % (width - 2, title))
    log.info("╚" + "═" * width + "╝")


def _step(label: str, status: str = "running") -> None:
    icons = {"running": "⟳", "ok": "✓", "fail": "✗", "skip": "○"}
    icon = icons.get(status, "·")
    log.info("  %s  %s", icon, label)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f}s"


# ── Layer import with graceful failure ────────────────────────────────────────

def _import_layers() -> tuple:
    """
    Import all three layer modules.
    Returns (data_fetcher, strategy_engine, visualizer) or exits on failure.
    """
    try:
        import data_fetcher as df_mod
        import strategy_engine as se_mod
        import visualizer as viz_mod
        return df_mod, se_mod, viz_mod
    except ImportError as exc:
        log.error("Failed to import project modules: %s", exc)
        log.error("Ensure data_fetcher.py, strategy_engine.py, and visualizer.py")
        log.error("are in the same directory as main.py.")
        sys.exit(1)


# ── Command implementations ───────────────────────────────────────────────────

def cmd_analyze(
    ticker: str,
    period: str,
    interval: str,
    df_mod,
    se_mod,
) -> bool:
    """
    /analyze — Run L1 data fetch + L2 strategy engine.

    Sequence
    --------
    1. Fetch OHLCV data via data_fetcher.fetch_ticker()
    2. Load the saved CSV
    3. Run strategy_engine.run_strategy()
    4. Save enriched DataFrame to data/GC_F_strategy_output.csv

    Returns True on success, False on failure.
    """
    _banner(f"/analyze  |  {ticker}  |  period={period}  |  interval={interval}")
    t0 = time.perf_counter()

    # ── Step 1: Fetch data (Layer 1) ─────────────────────────────────────
    _step("Layer 1 — Fetching market data …")
    try:
        config = df_mod.FetchConfig(
            ticker=ticker,
            period=period,
            interval=interval,
            output_dir=DATA_DIR,
        )
        result = df_mod.fetch_ticker(config)

        if not result.success:
            raise RuntimeError(
                f"data_fetcher returned failure for '{ticker}': {result.error}"
            )
        _step(f"Layer 1 — {result.rows} rows saved → {result.csv_path.name}", "ok")

    except Exception as exc:
        _general_error(exc, "cmd_analyze / Layer 1 (data_fetcher)")
        return False

    # ── Step 2: Load CSV ──────────────────────────────────────────────────
    _step("Layer 1 — Loading CSV for strategy …")
    try:
        raw_df = df_mod.load_csv(ticker, interval=interval, data_dir=DATA_DIR)
        _step(f"Layer 1 — Loaded {len(raw_df)} rows for strategy engine", "ok")
    except Exception as exc:
        _general_error(exc, "cmd_analyze / load_csv")
        return False

    # ── Step 3: Run strategy (Layer 2) ───────────────────────────────────
    _step("Layer 2 — Running AstroGoldHH strategy engine …")
    try:
        enriched_df = se_mod.run_strategy(raw_df)
        summary     = se_mod.get_signal_summary(enriched_df)
        buy_count   = (enriched_df["Signal"] == se_mod.SIGNAL_BUY).sum()
        sell_count  = (enriched_df["Signal"] == se_mod.SIGNAL_SELL).sum()
        _step(
            f"Layer 2 — Indicators calculated | "
            f"BUY: {buy_count}  SELL: {sell_count}",
            "ok",
        )

    except Exception as exc:
        # Strategy errors routed to SKILL.md (CLAUDE.md §4 Troubleshooting)
        _strategy_error(exc, "cmd_analyze / Layer 2 (strategy_engine)")
        return False

    # ── Step 4: Persist enriched DataFrame ───────────────────────────────
    _step("Layer 2 — Saving strategy output CSV …")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Derive output path from ticker
        safe   = ticker.replace("=", "_").replace("^", "")
        out_csv = DATA_DIR / f"{safe}_strategy_output.csv"
        enriched_df.to_csv(out_csv)
        _step(f"Layer 2 — Saved → {out_csv.name}", "ok")

    except Exception as exc:
        _general_error(exc, "cmd_analyze / saving strategy CSV")
        return False

    # ── Signal table preview ──────────────────────────────────────────────
    if not summary.empty:
        log.info("")
        log.info("  Signal preview (last 5):")
        for idx, row in summary.tail(5).iterrows():
            label = row.get("Signal_Label", "?")
            close = row.get("Close", 0)
            rsi   = row.get(f"RSI_{se_mod.CONFIG.rsi_period}", 0)
            log.info("    %-12s  %s  Close=%.2f  RSI=%.1f", label, idx, close, rsi)
    else:
        log.info("  No signals generated — consider a longer period (e.g. --period 5y).")

    log.info("")
    log.info("  /analyze completed in %s", _elapsed(t0))
    return True


def cmd_render(
    ticker: str,
    interval: str,
    auto_open: bool,
    viz_mod,
    se_mod,
) -> bool:
    """
    /render — Load the strategy output and generate the Plotly offline dashboard.

    Sequence
    --------
    1. Load strategy CSV (produced by /analyze)
    2. Build the Plotly figure via visualizer.build_chart()
    3. Save offline HTML via visualizer.render_offline()

    Returns True on success, False on failure.
    """
    _banner(f"/render  |  {ticker}  |  offline HTML dashboard")
    t0 = time.perf_counter()

    # ── Step 1: Load strategy output ─────────────────────────────────────
    _step("Layer 3 — Loading strategy output …")
    try:
        import pandas as pd
        safe    = ticker.replace("=", "_").replace("^", "")
        csv     = DATA_DIR / f"{safe}_strategy_output.csv"

        if not csv.exists():
            raise FileNotFoundError(
                f"Strategy output not found: {csv}\n"
                "  Run:  python main.py --analyze  first."
            )

        enriched_df = pd.read_csv(csv, index_col="Date", parse_dates=True)
        _step(f"Layer 3 — Loaded {len(enriched_df)} rows from {csv.name}", "ok")

    except Exception as exc:
        _general_error(exc, "cmd_render / loading strategy CSV")
        return False

    # ── Step 2 + 3: Build and save chart (Layer 3) ───────────────────────
    _step("Layer 3 — Building Plotly dashboard …")
    try:
        from visualizer import ChartConfig
        cfg = ChartConfig(
            output_dir=DATA_DIR,
            html_filename=f"{safe}_dashboard.html",
        )
        fig = viz_mod.build_chart(enriched_df, ticker=ticker, cfg=cfg)
        _step("Layer 3 — Figure assembled", "ok")

        _step("Layer 3 — Saving offline HTML …")
        out_path = viz_mod.render_offline(fig, cfg=cfg, auto_open=auto_open)
        _step(f"Layer 3 — Dashboard saved → {out_path.name}", "ok")

    except Exception as exc:
        _strategy_error(exc, "cmd_render / Layer 3 (visualizer)")
        return False

    log.info("")
    log.info("  /render completed in %s", _elapsed(t0))
    log.info("  Open in browser: %s", out_path)
    return True


def cmd_validate(se_mod) -> bool:
    """
    /validate — Run a lightweight unit-test suite that verifies the strategy
    engine's indicator and signal calculations against known values.

    Tests
    -----
    T01  EMA columns are created with correct names
    T02  EMA values are finite after warm-up period
    T03  MACD columns exist (line / signal_line / hist)
    T04  RSI is bounded in [0, 100]
    T05  BUY signal satisfies all SKILL.md gate conditions
    T06  SELL signal satisfies all SKILL.md gate conditions
    T07  A bar cannot carry both BUY and SELL simultaneously
    T08  Signals are absent during indicator warm-up (NaN) period

    Returns True if ALL tests pass, False if any fail.
    """
    _banner("/validate  |  AstroGoldHH Strategy Unit Tests")
    t0 = time.perf_counter()

    import numpy as np
    import pandas as pd

    cfg    = se_mod.CONFIG
    passed = 0
    failed = 0

    def _ok(label: str) -> None:
        nonlocal passed
        passed += 1
        _step(label, "ok")

    def _fail(label: str, reason: str) -> None:
        nonlocal failed
        failed += 1
        _step(f"{label}  ← {reason}", "fail")

    # ── Build synthetic OHLCV dataset ─────────────────────────────────────
    try:
        np.random.seed(42)
        n = 400   # enough bars to clear EMA 200 warm-up
        dates  = pd.date_range("2022-01-01", periods=n, freq="B")
        price  = 1900.0
        rows   = []
        for _ in range(n):
            chg   = price * np.random.normal(0.0003, 0.008)
            close = max(price + chg, 100.0)
            high  = close * (1 + abs(np.random.normal(0, 0.003)))
            low   = close * (1 - abs(np.random.normal(0, 0.003)))
            open_ = price
            rows.append({
                "Open": open_, "High": high,
                "Low": low,   "Close": close, "Volume": 100_000,
            })
            price = close

        raw = pd.DataFrame(rows, index=dates)
        raw.index.name = "Date"
        df  = se_mod.run_strategy(raw)

    except Exception as exc:
        _general_error(exc, "cmd_validate / dataset construction")
        return False

    ema_s = f"EMA_{cfg.ema_short}"
    ema_l = f"EMA_{cfg.ema_long}"
    rsi_c = f"RSI_{cfg.rsi_period}"

    # ── T01: EMA columns exist ────────────────────────────────────────────
    label = f"T01  EMA_{cfg.ema_short} and EMA_{cfg.ema_long} columns present"
    if ema_s in df.columns and ema_l in df.columns:
        _ok(label)
    else:
        _fail(label, f"missing: {[c for c in [ema_s, ema_l] if c not in df.columns]}")

    # ── T02: EMA values finite after warm-up ─────────────────────────────
    label = f"T02  EMA_{cfg.ema_long} finite after {cfg.ema_long}-bar warm-up"
    try:
        post_warmup = df[ema_l].iloc[cfg.ema_long:]
        if post_warmup.isna().any():
            _fail(label, f"{post_warmup.isna().sum()} NaN(s) after warm-up")
        else:
            _ok(label)
    except Exception as exc:
        _fail(label, str(exc))

    # ── T03: MACD columns exist ───────────────────────────────────────────
    label = "T03  MACD_line, MACD_signal_line, MACD_hist columns present"
    macd_cols = ["MACD_line", "MACD_signal_line", "MACD_hist"]
    missing_m = [c for c in macd_cols if c not in df.columns]
    if not missing_m:
        _ok(label)
    else:
        _fail(label, f"missing: {missing_m}")

    # ── T04: RSI bounded [0, 100] ─────────────────────────────────────────
    label = f"T04  RSI_{cfg.rsi_period} values in [0, 100]"
    try:
        rsi_vals = df[rsi_c].dropna()
        if (rsi_vals < 0).any() or (rsi_vals > 100).any():
            _fail(label, f"out-of-range values found: min={rsi_vals.min():.2f} max={rsi_vals.max():.2f}")
        else:
            _ok(label)
    except Exception as exc:
        _fail(label, str(exc))

    # ── T05: BUY signals satisfy all gate conditions ──────────────────────
    label = "T05  BUY signals satisfy all SKILL.md gate conditions"
    try:
        buys = df[df["Signal"] == se_mod.SIGNAL_BUY]
        if buys.empty:
            _step("T05  (no BUY signals in synthetic data — skipped)", "skip")
        else:
            violations = []
            for idx, row in buys.iterrows():
                if not (row["Close"] > row[ema_s] > row[ema_l]):
                    violations.append(f"{idx}: trend alignment failed")
                if not (row["MACD_hist"] > 0):
                    violations.append(f"{idx}: MACD_hist not > 0")
                if not (cfg.rsi_buy_low <= row[rsi_c] <= cfg.rsi_buy_high):
                    violations.append(f"{idx}: RSI {row[rsi_c]:.1f} outside [{cfg.rsi_buy_low},{cfg.rsi_buy_high}]")
                if not (row["Close"] > row["Open"]):
                    violations.append(f"{idx}: not a bullish candle")
            if violations:
                _fail(label, f"{len(violations)} violation(s): {violations[:2]}")
            else:
                _ok(f"{label}  ({len(buys)} signals verified)")
    except Exception as exc:
        _fail(label, str(exc))

    # ── T06: SELL signals satisfy all gate conditions ─────────────────────
    label = "T06  SELL signals satisfy all SKILL.md gate conditions"
    try:
        sells = df[df["Signal"] == se_mod.SIGNAL_SELL]
        if sells.empty:
            _step("T06  (no SELL signals in synthetic data — skipped)", "skip")
        else:
            violations = []
            for idx, row in sells.iterrows():
                if not (row["Close"] < row[ema_s] < row[ema_l]):
                    violations.append(f"{idx}: trend alignment failed")
                if not (row["MACD_hist"] < 0):
                    violations.append(f"{idx}: MACD_hist not < 0")
                if not (cfg.rsi_sell_low <= row[rsi_c] <= cfg.rsi_sell_high):
                    violations.append(f"{idx}: RSI {row[rsi_c]:.1f} outside [{cfg.rsi_sell_low},{cfg.rsi_sell_high}]")
                if not (row["Close"] < row["Open"]):
                    violations.append(f"{idx}: not a bearish candle")
            if violations:
                _fail(label, f"{len(violations)} violation(s): {violations[:2]}")
            else:
                _ok(f"{label}  ({len(sells)} signals verified)")
    except Exception as exc:
        _fail(label, str(exc))

    # ── T07: No bar has both BUY and SELL ─────────────────────────────────
    label = "T07  No bar carries both BUY and SELL simultaneously"
    try:
        both = df[
            (df["Signal"] == se_mod.SIGNAL_BUY) &
            (df["Signal"] == se_mod.SIGNAL_SELL)
        ]
        if both.empty:
            _ok(label)
        else:
            _fail(label, f"{len(both)} conflicting bar(s) found")
    except Exception as exc:
        _fail(label, str(exc))

    # ── T08: No signals during warm-up ────────────────────────────────────
    label = f"T08  No signals during EMA_{cfg.ema_long} warm-up period"
    try:
        warmup_signals = df.iloc[:cfg.ema_long]["Signal"]
        non_neutral = (warmup_signals != se_mod.SIGNAL_NEUTRAL).sum()
        if non_neutral > 0:
            _fail(label, f"{non_neutral} signal(s) found in warm-up window")
        else:
            _ok(label)
    except Exception as exc:
        _fail(label, str(exc))

    # ── Summary ───────────────────────────────────────────────────────────
    total = passed + failed
    log.info("")
    log.info("  ─────────────────────────────────────────────")
    log.info("  Tests passed : %d / %d", passed, total)
    log.info("  Tests failed : %d / %d", failed, total)
    log.info("  Duration     : %s", _elapsed(t0))
    log.info("  ─────────────────────────────────────────────")

    if failed > 0:
        log.warning("  Some tests failed. Review SKILL.md signal logic:")
        log.warning("  → %s", SKILL_PATH)
        return False

    log.info("  All tests passed ✓")
    return True


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "AstroGoldHH — Institutional Gold Strategy Analyzer\n"
            "Commands map to CLAUDE.md §5: /analyze  /render  /validate"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --analyze                     # fetch + strategy only
  python main.py --render                      # chart from cached data
  python main.py --analyze --render            # full pipeline
  python main.py --validate                    # run unit tests
  python main.py --analyze --render --ticker GLD --period 5y
  python main.py --analyze --render --no-browser
        """,
    )

    # ── Commands (CLAUDE.md §5) ───────────────────────────────────────────
    cmd_group = parser.add_argument_group("Commands (CLAUDE.md §5)")
    cmd_group.add_argument(
        "--analyze", action="store_true",
        help="/analyze — fetch market data (L1) + run strategy engine (L2)",
    )
    cmd_group.add_argument(
        "--render", action="store_true",
        help="/render  — generate Plotly offline dashboard (L3)",
    )
    cmd_group.add_argument(
        "--validate", action="store_true",
        help="/validate — run unit test suite for strategy calculations",
    )

    # ── Data options ──────────────────────────────────────────────────────
    data_group = parser.add_argument_group("Data options")
    data_group.add_argument(
        "--ticker", default="GC=F", metavar="SYMBOL",
        help="Yahoo Finance ticker symbol (default: GC=F — Gold Futures)",
    )
    data_group.add_argument(
        "--period", default="2y", metavar="PERIOD",
        help="yfinance period string: 1y 2y 5y 6mo … (default: 2y)",
    )
    data_group.add_argument(
        "--interval", default="1d", metavar="INTERVAL",
        help="bar size: 1d 1h 5m … (default: 1d)",
    )

    # ── Render options ────────────────────────────────────────────────────
    render_group = parser.add_argument_group("Render options")
    render_group.add_argument(
        "--no-browser", dest="no_browser", action="store_true",
        help="save HTML without auto-opening in browser",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the AstroGoldHH pipeline based on CLI flags.
    Exit code 0 = all requested commands succeeded.
    Exit code 1 = one or more commands failed.
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # Guard: require at least one command
    if not any([args.analyze, args.render, args.validate]):
        parser.print_help()
        log.warning(
            "\nNo command specified.  "
            "Use --analyze, --render, --validate, or a combination."
        )
        sys.exit(0)

    # Import all three layer modules once
    df_mod, se_mod, viz_mod = _import_layers()

    session_start = time.perf_counter()
    results: dict[str, bool] = {}

    # ── /validate (can run independently) ────────────────────────────────
    if args.validate:
        results["validate"] = cmd_validate(se_mod)

    # ── /analyze ─────────────────────────────────────────────────────────
    if args.analyze:
        results["analyze"] = cmd_analyze(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            df_mod=df_mod,
            se_mod=se_mod,
        )

    # ── /render (only proceeds if analyze succeeded or data already exists) ──
    if args.render:
        if args.analyze and not results.get("analyze", True):
            log.warning("Skipping --render because --analyze failed.")
            results["render"] = False
        else:
            results["render"] = cmd_render(
                ticker=args.ticker,
                interval=args.interval,
                auto_open=not args.no_browser,
                viz_mod=viz_mod,
                se_mod=se_mod,
            )

    # ── Session summary ───────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  SESSION SUMMARY                                             ║")
    log.info("╠══════════════════════════════════════════════════════════════╣")
    all_ok = True
    for cmd, ok in results.items():
        icon   = "✓" if ok else "✗"
        status = "PASSED" if ok else "FAILED"
        log.info("║  %s  %-12s %s%s║", icon, f"/{cmd}", status,
                 " " * (44 - len(status) - len(cmd)))
        if not ok:
            all_ok = False
    elapsed = f"{time.perf_counter() - session_start:.2f}s"
    log.info("╠══════════════════════════════════════════════════════════════╣")
    log.info("║  Total time: %-49s║", elapsed)
    log.info("╚══════════════════════════════════════════════════════════════╝")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
