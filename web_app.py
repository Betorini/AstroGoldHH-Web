"""
AstroGoldHH — Web Interface (web_app.py)
==========================================
A Streamlit GUI that wraps the existing CLI project without modifying it.

Architecture decision
---------------------
This file is a BRIDGE only. It imports pure functions from the existing
layer modules and calls them directly — it never calls main() or any
function that invokes sys.exit().

Safe import map:
    data_fetcher    → fetch_ticker(), load_csv(), FetchConfig()     (Layer 1)
    strategy_engine → run_strategy(), get_signal_summary(),         (Layer 2)
                      CONFIG, SIGNAL_BUY, SIGNAL_SELL
    visualizer      → build_chart(), ChartConfig()                  (Layer 3)
    main            → cmd_analyze(), cmd_render(), cmd_validate()   (Layer 4)
                      ── these return bool, never call sys.exit() ──

NOT imported / NOT called:
    main.main()           → calls sys.exit() — would kill Streamlit
    main._import_layers() → calls sys.exit() on ImportError
    main._build_parser()  → argparse-only, irrelevant here

Coexistence guarantee
---------------------
    streamlit run web_app.py   → launches this web interface
    python main.py --analyze   → CLI still works identically, untouched

Run:
    streamlit run web_app.py
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Project root (allows running from any working directory) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR  = PROJECT_ROOT / "data"
LOG_DIR   = PROJECT_ROOT / "logs"
SKILL_PATH = PROJECT_ROOT / ".claude" / "skills" / "SKILL.md"


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  SECTION 1 — SAFE LAYER IMPORTS
# ║  Each import is isolated so a single missing module shows a clear error
# ║  instead of crashing the whole app.
# ╚══════════════════════════════════════════════════════════════════════════════

def _safe_import_layers() -> dict:
    """
    Import each project module independently.
    Returns a dict of module references; sets module to None on failure.
    Never calls sys.exit() — errors are surfaced via st.error() instead.
    """
    modules: dict = {
        "data_fetcher":    None,
        "strategy_engine": None,
        "visualizer":      None,
        "main":            None,
    }
    for name in modules:
        try:
            import importlib
            modules[name] = importlib.import_module(name)
        except ImportError as exc:
            st.error(
                f"**Module import failed:** `{name}.py` — {exc}\n\n"
                f"Ensure `{name}.py` is in the same directory as `web_app.py`."
            )
    return modules


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  SECTION 2 — WEB ADAPTER FUNCTIONS
# ║  Thin wrappers around cmd_* that catch all exceptions and return
# ║  structured results safe for Streamlit to consume.
# ╚══════════════════════════════════════════════════════════════════════════════

def web_analyze(
    ticker: str,
    period: str,
    interval: str,
    mods: dict,
) -> tuple[bool, Optional[pd.DataFrame], str]:
    """
    Web-safe wrapper around main.cmd_analyze().

    Calls cmd_analyze() which returns bool and never calls sys.exit().
    Also returns the enriched DataFrame directly so the web UI can render
    it immediately without re-loading from disk.

    Returns
    -------
    (success: bool, dataframe: DataFrame | None, message: str)
    """
    df_mod = mods["data_fetcher"]
    se_mod = mods["strategy_engine"]
    main_mod = mods["main"]

    if not all([df_mod, se_mod, main_mod]):
        return False, None, "One or more required modules failed to import."

    try:
        # ── Step 1: call the existing CLI function ────────────────────────
        success = main_mod.cmd_analyze(
            ticker=ticker,
            period=period,
            interval=interval,
            df_mod=df_mod,
            se_mod=se_mod,
        )

        if not success:
            return False, None, (
                f"Analysis failed for '{ticker}'. "
                "Check the ticker symbol and try a longer period."
            )

        # ── Step 2: load the enriched DataFrame from disk ─────────────────
        safe = ticker.replace("=", "_").replace("^", "")
        csv_path = DATA_DIR / f"{safe}_strategy_output.csv"

        if not csv_path.exists():
            return False, None, (
                f"Strategy CSV not found at {csv_path}. "
                "Analysis may have failed silently."
            )

        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

        buys  = (df["Signal"] == se_mod.SIGNAL_BUY).sum()
        sells = (df["Signal"] == se_mod.SIGNAL_SELL).sum()
        msg   = (
            f"✓ Analysis complete — {len(df)} bars loaded | "
            f"BUY signals: **{buys}** | SELL signals: **{sells}**"
        )
        return True, df, msg

    except SystemExit as exc:
        # Defensive: catch any unexpected sys.exit() from the layer modules
        # so Streamlit doesn't terminate.
        return False, None, (
            f"A module called sys.exit({exc.code}). "
            "This was intercepted safely. Check logs for details."
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _write_error_log("web_analyze", exc, tb)
        return False, None, f"Unexpected error during analysis: {exc}"


def web_render(
    ticker: str,
    interval: str,
    df: Optional[pd.DataFrame],
    mods: dict,
) -> tuple[bool, Optional[object], str]:
    """
    Web-safe wrapper around main.cmd_render() and visualizer.build_chart().

    Instead of saving an HTML file (offline mode), we build the Plotly
    figure and return it directly for st.plotly_chart() — keeping the
    render 100% in-memory and cloud-safe.

    Returns
    -------
    (success: bool, plotly_figure | None, message: str)
    """
    viz_mod = mods["visualizer"]
    se_mod  = mods["strategy_engine"]

    if not all([viz_mod, se_mod]):
        return False, None, "Visualizer or strategy_engine module not available."

    if df is None:
        return False, None, "No data available. Run /analyze first."

    try:
        from visualizer import ChartConfig  # type: ignore[import]
        cfg = ChartConfig(
            title=f"AstroGoldHH — {ticker}",
            output_dir=DATA_DIR,
            html_filename=f"{ticker.replace('=','_')}_dashboard.html",
        )
        fig = viz_mod.build_chart(df, ticker=ticker, cfg=cfg)
        return True, fig, f"✓ Chart rendered for **{ticker}**"

    except SystemExit as exc:
        return False, None, f"sys.exit({exc.code}) intercepted during render."
    except Exception as exc:
        tb = traceback.format_exc()
        _write_error_log("web_render", exc, tb)
        return False, None, (
            f"Chart build failed: {exc}\n\n"
            f"Consult strategy reference: `{SKILL_PATH}`"
        )


def web_validate(mods: dict) -> tuple[bool, list[dict]]:
    """
    Web-safe wrapper around main.cmd_validate().

    cmd_validate() logs results via the logging module. We intercept those
    log records using a MemoryHandler so they can be displayed in Streamlit
    without relying on console output.

    Returns
    -------
    (all_passed: bool, test_results: list[dict])
    Each dict: {"label": str, "passed": bool | None, "skipped": bool}
    """
    se_mod   = mods["strategy_engine"]
    main_mod = mods["main"]

    if not all([se_mod, main_mod]):
        return False, [{"label": "Module import failed", "passed": False, "skipped": False}]

    # Intercept log records produced by cmd_validate()
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger   = logging.getLogger("AstroGoldHH")
    handler  = _Capture()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    try:
        all_passed = main_mod.cmd_validate(se_mod)
    except SystemExit as exc:
        logger.removeHandler(handler)
        return False, [{"label": f"sys.exit({exc.code}) intercepted", "passed": False, "skipped": False}]
    except Exception as exc:
        logger.removeHandler(handler)
        return False, [{"label": f"Exception: {exc}", "passed": False, "skipped": False}]
    finally:
        logger.removeHandler(handler)

    # Parse log records into structured test results
    results: list[dict] = []
    for rec in records:
        msg = rec.getMessage()
        # Only keep test lines (start with T0x)
        if not any(msg.strip().startswith(f"T{i:02d}") for i in range(1, 20)):
            continue
        skipped = "(no " in msg.lower() or "skipped" in msg.lower()
        passed  = None if skipped else ("✓" in msg or "ok" in msg.lower())
        results.append({"label": msg.strip(), "passed": passed, "skipped": skipped})

    return all_passed, results


def _write_error_log(context: str, exc: Exception, tb: str) -> None:
    """Write error details to logs/ so the user can investigate."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        log_file = LOG_DIR / f"web_app_{datetime.now():%Y%m%d}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] ERROR in {context}\n")
            f.write(tb)
    except Exception:
        pass  # Never let logging crash the app


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  SECTION 3 — UI HELPERS
# ╚══════════════════════════════════════════════════════════════════════════════

PRESET_TICKERS = [
    "GC=F", "GLD", "SI=F", "SPY", "QQQ",
    "AAPL", "MSFT", "TSLA", "NVDA", "^DXY", "BTC-USD",
]
PERIOD_OPTIONS  = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
INTERVAL_OPTIONS = {"Daily": "1d", "Weekly": "1wk"}


def _render_kpi_row(df: pd.DataFrame, ticker: str, se_mod) -> None:
    """Top metrics strip — latest price, return, RSI, signal counts."""
    close     = df["Close"].dropna()
    latest    = close.iloc[-1]
    prev      = close.iloc[-2] if len(close) > 1 else latest
    chg_pct   = (latest / prev - 1) * 100
    period_rt = (close.iloc[-1] / close.iloc[0] - 1) * 100
    rsi_col   = f"RSI_{se_mod.CONFIG.rsi_period}"
    rsi_val   = df[rsi_col].dropna().iloc[-1] if rsi_col in df.columns else float("nan")
    buys      = int((df["Signal"] == se_mod.SIGNAL_BUY).sum())
    sells     = int((df["Signal"] == se_mod.SIGNAL_SELL).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{ticker} Close",    f"${latest:,.2f}", f"{chg_pct:+.2f}%")
    c2.metric("Period Return",      f"{period_rt:+.2f}%")
    c3.metric("RSI (current)",      f"{rsi_val:.1f}")
    c4.metric("BUY Signals",        buys)
    c5.metric("SELL Signals",       sells, delta_color="inverse")


def _render_signal_table(df: pd.DataFrame, se_mod) -> None:
    """Date-stamped signal log table."""
    cfg     = se_mod.CONFIG
    rsi_col = f"RSI_{cfg.rsi_period}"
    ema_s   = f"EMA_{cfg.ema_short}"
    ema_l   = f"EMA_{cfg.ema_long}"

    signals = df[df["Signal"] != se_mod.SIGNAL_NEUTRAL].copy()
    if signals.empty:
        st.info(
            "No signals detected. Try a longer period (e.g. 5 Years) "
            "or a ticker with clearer trend structure."
        )
        return

    signals["Type"] = signals["Signal"].map(
        {se_mod.SIGNAL_BUY: "🟢 BUY", se_mod.SIGNAL_SELL: "🔴 SELL"}
    )
    keep     = [c for c in ["Type", "Close", ema_s, ema_l, "MACD_hist", rsi_col]
                if c in signals.columns]
    out      = signals[keep].copy()
    out.index = out.index.strftime("%Y-%m-%d")
    out       = out.rename(columns={
        ema_s: f"EMA {cfg.ema_short}", ema_l: f"EMA {cfg.ema_long}",
        "MACD_hist": "MACD Hist", rsi_col: f"RSI {cfg.rsi_period}",
    })

    def _row_color(row):
        c = "rgba(63,185,80,0.12)" if "BUY" in str(row["Type"]) else "rgba(248,81,73,0.12)"
        return [f"background-color: {c}"] * len(row)

    fmt = {col: "{:.2f}" for col in out.columns if col != "Type"}
    st.dataframe(
        out.style.apply(_row_color, axis=1).format(fmt),
        use_container_width=True,
        height=min(420, 38 * (len(out) + 1)),
    )


def _render_user_manual() -> None:
    """Bilingual User Manual — Thai + English inside a sidebar expander."""
    with st.expander("📖 User Manual (คู่มือการใช้งาน)"):

        # ── ภาษาไทย ──────────────────────────────────────────────────────
        st.markdown("### 🇹🇭 คู่มือการใช้งาน")

        st.markdown("**📌 การเลือก Ticker (สัญลักษณ์หุ้น/สินทรัพย์)**")
        st.markdown("""
- เลือกสัญลักษณ์จากรายการที่กำหนดไว้ เช่น `GC=F` (ทองคำ Futures), `GLD` (Gold ETF)
- หรือพิมพ์ชื่อ Ticker ที่ต้องการในช่อง "Or enter custom ticker"
- รองรับสัญลักษณ์จาก Yahoo Finance เช่น `AAPL`, `BTC-USD`, `^DXY`
        """)

        st.markdown("**📅 การเลือก Timeframe (กรอบเวลา)**")
        st.markdown("""
- **Period:** เลือกช่วงเวลาย้อนหลัง ตั้งแต่ 6 เดือน ถึง 5 ปี
- **Interval:** ขนาด Candle — Daily (รายวัน) หรือ Weekly (รายสัปดาห์)
- แนะนำ **2 ปี / Daily** สำหรับ EMA 200 ที่มีข้อมูลเพียงพอ
        """)

        st.markdown("**🔬 Strategy Config (การตั้งค่ากลยุทธ์)**")
        st.markdown("""
- **EMA 50/200:** เส้นแนวโน้มระยะสั้น/ยาว — BUY เมื่อ ราคา > EMA50 > EMA200
- **MACD (12/26/9):** วัดโมเมนตัม — Histogram > 0 = แรงซื้อยืนยัน
- **RSI (14):** ช่วง 50–65 สัญญาณซื้อ | ช่วง 35–50 สัญญาณขาย
- ค่าทั้งหมดตาม J.P. Morgan Institutional Methodology
        """)

        st.markdown("**📈 การอ่านกราฟ**")
        st.markdown("""
- **แถวบน:** แท่งเทียน + EMA 50 (น้ำเงิน) + EMA 200 (เหลือง)
- **🟢 ▲ (Triangle Up):** สัญญาณ BUY — เงื่อนไขทั้ง 5 ข้อผ่านครบในแท่งเดียว
- **🔴 ▼ (Triangle Down):** สัญญาณ SELL — เงื่อนไข Bearish ครบทั้ง 5 ข้อ
- **แถวกลาง:** MACD Histogram + เส้น MACD และ Signal
- **แถวล่าง:** RSI พร้อมแถบโซนซื้อ (เขียว) และโซนขาย (แดง)
        """)

        st.markdown("**📋 Summary Metrics**")
        st.markdown("""
- **Latest Close:** ราคาปิดล่าสุด และ % เปลี่ยนแปลงจากวันก่อน
- **Period Return:** ผลตอบแทนรวมตลอดช่วงเวลาที่เลือก
- **RSI (current):** ค่า RSI ปัจจุบัน — ช่วยระบุโซนตลาด
- **BUY/SELL Signals:** จำนวนสัญญาณที่ตรวจพบทั้งหมด
        """)

        st.markdown("**🖥️ CLI Commands (สำหรับ Developer)**")
        st.markdown("""
แอปนี้ยังคงรองรับการใช้งานผ่าน Terminal:
```bash
python main.py --analyze --ticker GC=F --period 2y
python main.py --render  --no-browser
python main.py --validate
python main.py --analyze --render --ticker GLD --period 5y
```
        """)

        st.divider()

        # ── English ───────────────────────────────────────────────────────
        st.markdown("### 🇬🇧 User Manual")

        st.markdown("**📌 Ticker Selection**")
        st.markdown("""
- Choose from the preset list (`GC=F`, `GLD`, `SPY`, `BTC-USD`, etc.)
- Or type any valid Yahoo Finance symbol in the custom ticker field
- Examples: `AAPL`, `NVDA`, `XAUUSD=X`, `^DXY`
        """)

        st.markdown("**📅 Timeframe**")
        st.markdown("""
- **Period:** Lookback window — 6 months to 5 years
- **Interval:** Candle size — Daily or Weekly
- Recommended: **2 Years / Daily** to give EMA 200 sufficient warm-up data
        """)

        st.markdown("**🔬 Strategy Config**")
        st.markdown("""
- **EMA 50/200:** Trend filter — Bullish when Close > EMA 50 > EMA 200
- **MACD (12/26/9):** Momentum gate — Histogram > 0 confirms bullish momentum
- **RSI (14):** Oscillator gate — BUY zone: 50–65 | SELL zone: 35–50
- All values follow the **AstroGoldHH Institutional Strategy** (J.P. Morgan methodology)
        """)

        st.markdown("**📈 Reading the Chart**")
        st.markdown("""
- **Top panel:** Candlestick + EMA 50 (blue) + EMA 200 (amber)
- **🟢 ▲ BUY marker:** All 5 gates pass simultaneously — bullish trend, MACD > 0, RSI 50–65, candle touches EMA 50, green reversal candle
- **🔴 ▼ SELL marker:** All 5 bearish gates pass — bearish trend, MACD < 0, RSI 35–50, candle tests EMA 50 as resistance, red reversal candle
- **Middle panel:** MACD histogram (green/red bars) + MACD line + Signal line
- **Bottom panel:** RSI with shaded buy zone (green 50–65) and sell zone (red 35–50)
- **Hover** anywhere on the chart for exact OHLC, RSI, and signal values
        """)

        st.markdown("**📋 Summary Metrics**")
        st.markdown("""
- **Latest Close:** Most recent price with daily % change
- **Period Return:** Total return over the selected lookback window
- **RSI (current):** Latest RSI reading — useful for quick zone assessment
- **BUY / SELL Signals:** Count of all signals detected in the period
- Scroll down to the **Signal Log** for the full date-by-date table
        """)

        st.markdown("**🖥️ CLI — still works perfectly**")
        st.markdown("""
This web UI is a separate file. Your CLI is untouched:
```bash
python main.py --analyze --ticker GC=F --period 2y
python main.py --render  --no-browser
python main.py --validate
```
        """)

        st.caption("⚠️ For informational purposes only. Not financial advice.")


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  SECTION 4 — MAIN STREAMLIT APPLICATION
# ╚══════════════════════════════════════════════════════════════════════════════

def _render_sidebar(mods: dict, cfg) -> tuple[str, str, str, str, bool, bool, bool]:
    """
    Render all sidebar widgets and return user selections + button states.
    Returns: (ticker, period, interval, period_label, btn_analyze, btn_render, btn_validate)
    """
    with st.sidebar:
        st.title("⚙️ AstroGoldHH")
        st.caption("Institutional Strategy · Web Interface")
        st.divider()

        st.subheader("📌 Ticker")
        ticker_preset = st.selectbox(
            "Preset", PRESET_TICKERS, index=0, label_visibility="collapsed"
        )
        ticker_custom = st.text_input("Custom ticker", placeholder="e.g. XAUUSD=X, GDX …")
        ticker = ticker_custom.strip().upper() if ticker_custom.strip() else ticker_preset

        st.subheader("📅 Timeframe")
        period_label   = st.selectbox("Period",   list(PERIOD_OPTIONS.keys()),   index=1)
        interval_label = st.selectbox("Interval", list(INTERVAL_OPTIONS.keys()), index=0)
        period   = PERIOD_OPTIONS[period_label]
        interval = INTERVAL_OPTIONS[interval_label]

        if cfg:
            st.divider()
            st.subheader("🔬 Strategy Config")
            st.caption("AstroGoldHH-Institutional-Strategy")
            st.markdown(f"""
| Parameter | Value |
|---|---|
| EMA Short | {cfg.ema_short} |
| EMA Long  | {cfg.ema_long}  |
| MACD      | {cfg.macd_fast}/{cfg.macd_slow}/{cfg.macd_signal} |
| RSI       | {cfg.rsi_period} |
| BUY RSI   | [{cfg.rsi_buy_low} – {cfg.rsi_buy_high}] |
| SELL RSI  | [{cfg.rsi_sell_low} – {cfg.rsi_sell_high}] |
            """)

        # ── Action buttons — call existing CLI cmd_* functions ────────────
        st.divider()
        st.subheader("▶ Commands")
        st.caption("Each button calls the same function as the CLI command")
        col_a, col_v = st.columns(2)
        btn_analyze  = col_a.button(
            "📊 /analyze", use_container_width=True,
            help="Fetch data (L1) + strategy engine (L2) · same as: python main.py --analyze"
        )
        btn_validate = col_v.button(
            "✅ /validate", use_container_width=True,
            help="Run unit tests · same as: python main.py --validate"
        )
        btn_render = st.button(
            "🖼️ /render  (in-page)", use_container_width=True, type="primary",
            help="Build interactive chart · data must be loaded first"
        )

        st.divider()
        _render_user_manual()

        st.caption("📖 J.P. Morgan Institutional Methodology")
        st.caption("📊 Yahoo Finance · yfinance")
        st.caption(f"🖥️ CLI: `python main.py --analyze --ticker {ticker}`")

    return ticker, period, interval, period_label, btn_analyze, btn_render, btn_validate


def _render_main_area(
    ticker: str, period: str, interval: str, period_label: str,
    btn_analyze: bool, btn_render: bool, btn_validate: bool,
    mods: dict, cfg,
) -> None:
    """Render all main-area content: header, command results, chart, tables."""
    se_mod = mods["strategy_engine"]

    st.title("📈 AstroGoldHH — Institutional Gold Strategy")
    if cfg:
        st.caption(
            f"Trend-Following & Mean-Reversion  ·  "
            f"EMA {cfg.ema_short}/{cfg.ema_long}  ·  "
            f"MACD ({cfg.macd_fast},{cfg.macd_slow},{cfg.macd_signal})  ·  "
            f"RSI {cfg.rsi_period}"
        )

    # ── /validate ────────────────────────────────────────────────────────
    if btn_validate:
        st.subheader("✅ /validate — Strategy Unit Tests")
        with st.spinner("Running validation suite …"):
            t0 = time.perf_counter()
            all_passed, results = web_validate(mods)
            elapsed = f"{time.perf_counter() - t0:.2f}s"
        for r in results:
            if r["skipped"]:    st.info(f"  ○  {r['label']}")
            elif r["passed"]:   st.success(f"  ✓  {r['label']}")
            else:               st.error(f"  ✗  {r['label']}")
        if all_passed:
            st.success(f"**All tests passed** in {elapsed}")
        else:
            st.error(f"**Some tests failed** ({elapsed}).  Consult: `{SKILL_PATH}`")
        st.divider()

    # ── /analyze ─────────────────────────────────────────────────────────
    if btn_analyze:
        with st.spinner(f"Running /analyze for {ticker} ({period_label}) …"):
            t0 = time.perf_counter()
            ok, df, msg = web_analyze(ticker, period, interval, mods)
            elapsed = f"{time.perf_counter() - t0:.2f}s"
        if ok and df is not None:
            st.success(f"{msg}  ·  {elapsed}")
            st.session_state.update({
                "df": df, "ticker": ticker,
                "period": period_label, "interval": interval,
                "fig": None,   # invalidate stale figure
            })
        else:
            st.error(f"**/analyze failed:** {msg}")
            st.info("💡 Verify ticker · Try longer period · Check data_fetcher.py location")

    # ── /render ──────────────────────────────────────────────────────────
    if btn_render:
        df_now     = st.session_state.get("df")
        tkr_now    = st.session_state.get("ticker", ticker)
        if df_now is None:
            st.warning("⚠️  No data loaded — click **📊 /analyze** first.")
        else:
            with st.spinner("Building interactive chart …"):
                t0 = time.perf_counter()
                ok, fig, msg = web_render(tkr_now, interval, df_now, mods)
                elapsed = f"{time.perf_counter() - t0:.2f}s"
            if ok and fig is not None:
                st.session_state["fig"] = fig
                st.caption(f"{msg}  ·  {elapsed}")
            else:
                st.error(f"**/render failed:** {msg}")

    # Auto-render when analyze just ran and produced a fresh df
    if "df" in st.session_state and st.session_state.get("fig") is None:
        ok, fig, _ = web_render(
            st.session_state.get("ticker", ticker),
            interval, st.session_state["df"], mods,
        )
        if ok and fig:
            st.session_state["fig"] = fig

    # ── KPI + Chart + Tables ──────────────────────────────────────────────
    if "df" in st.session_state and se_mod:
        _render_kpi_row(st.session_state["df"], st.session_state.get("ticker", ticker), se_mod)
        st.divider()

    if st.session_state.get("fig") is not None:
        st.plotly_chart(
            st.session_state["fig"], use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True,
                    "displaylogo": False, "modeBarButtonsToAdd": ["drawline", "eraseshape"]},
        )
    elif "df" not in st.session_state:
        st.info("👈  Select a ticker and timeframe, then click **📊 /analyze**.")
        st.markdown("""
| Sidebar button | Equivalent CLI command |
|---|---|
| 📊 /analyze | `python main.py --analyze --ticker GC=F --period 2y` |
| 🖼️ /render  | `python main.py --render --no-browser` |
| ✅ /validate | `python main.py --validate` |
        """)

    if "df" in st.session_state and se_mod:
        tckr  = st.session_state.get("ticker", ticker)
        prd   = st.session_state.get("period", period_label)
        intv  = st.session_state.get("interval", interval)
        buys  = int((st.session_state["df"]["Signal"] == se_mod.SIGNAL_BUY).sum())
        sells = int((st.session_state["df"]["Signal"] == se_mod.SIGNAL_SELL).sum())
        st.subheader("📋 Signal Log")
        st.caption(f"{tckr} · {prd} · {intv} · **{buys} BUY** · **{sells} SELL**")
        _render_signal_table(st.session_state["df"], se_mod)

    if "df" in st.session_state and cfg:
        with st.expander("🔍 Raw data preview (last 50 bars)"):
            show  = ["Open", "High", "Low", "Close", "Volume",
                     f"EMA_{cfg.ema_short}", f"EMA_{cfg.ema_long}",
                     "MACD_hist", f"RSI_{cfg.rsi_period}", "Signal"]
            avail = [c for c in show if c in st.session_state["df"].columns]
            prev  = st.session_state["df"][avail].tail(50).copy()
            prev.index = prev.index.strftime("%Y-%m-%d")
            st.dataframe(
                prev.style.format({c: "{:.2f}" for c in avail if c not in ("Volume","Signal")}),
                use_container_width=True,
            )


def main() -> None:
    """Streamlit entry point — web_app's own main(), NOT main.py's."""
    st.set_page_config(
        page_title="AstroGoldHH — Institutional Strategy",
        page_icon="📈", layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown("""<style>
        .block-container{padding-top:1.5rem}
        [data-testid="stMetricValue"]{font-size:1.15rem;font-family:monospace}
        .stDataFrame{font-family:monospace;font-size:.85rem}
    </style>""", unsafe_allow_html=True)

    if "mods" not in st.session_state:
        with st.spinner("Loading project modules …"):
            st.session_state["mods"] = _safe_import_layers()

    mods   = st.session_state["mods"]
    se_mod = mods["strategy_engine"]
    cfg    = se_mod.CONFIG if se_mod else None

    ticker, period, interval, period_label, btn_a, btn_r, btn_v = _render_sidebar(mods, cfg)
    _render_main_area(ticker, period, interval, period_label, btn_a, btn_r, btn_v, mods, cfg)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
