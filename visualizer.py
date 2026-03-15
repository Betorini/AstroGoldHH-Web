"""
AstroGoldHH — Layer 3: Visualizer
====================================
Responsibility : Render an interactive, offline Plotly dashboard from the
                 signal-annotated DataFrame produced by Layer 2.

Architecture   : L4 Agents — this module has NO data-fetching logic and
                 NO strategy/indicator logic.  It only reads, layouts, and
                 renders.

SKILL.md ref   : §Visualization Requirements
                 - Main Chart : Candlesticks, EMA 50, EMA 200
                 - Subplot 1  : MACD lines + Histogram
                 - Subplot 2  : RSI with levels at 35, 50, 65
                 - Markers    : Green ▲ BUY  |  Red ▼ SELL

CLAUDE.md ref  : §3 Visualization
                 - Plotly must be used in offline mode
                 - Charts must be responsive and interactive
                 - Clearly mark Buy/Sell signals (AstroGoldHH standard)

Dependencies   : plotly, pandas
                 Install: pip install plotly pandas
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Optional dependency guard ────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
except ImportError:
    print("[FATAL] plotly is not installed.  Run:  pip install plotly")
    sys.exit(1)

# ── Layer 1 import (for load_csv convenience) ─────────────────────────────────
try:
    from data_fetcher import DATA_DIR, DEFAULT_INTERVAL, load_csv
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DEFAULT_INTERVAL = "1d"
    load_csv = None

# ── Layer 2 import ────────────────────────────────────────────────────────────
try:
    from strategy_engine import (
        CONFIG as STRATEGY_CONFIG,
        SIGNAL_BUY,
        SIGNAL_SELL,
        run_strategy,
    )
except ImportError:
    print("[FATAL] strategy_engine.py not found. Ensure it is in the same directory.")
    sys.exit(1)


# ── Logger ────────────────────────────────────────────────────────────────────

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("visualizer")
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
            log_dir / f"visualizer_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log: logging.Logger = _get_logger()


# ── Chart configuration ───────────────────────────────────────────────────────

@dataclass
class ChartConfig:
    """
    All visual styling constants in one place.
    Follows the AstroGoldHH standard marker spec from SKILL.md.
    """
    # Layout
    title: str              = "AstroGoldHH — Institutional Gold Strategy (GC=F)"
    output_dir: Path        = Path(__file__).resolve().parent / "data"
    html_filename: str      = "AstroGoldHH_dashboard.html"
    chart_height: int       = 1000

    # Row height ratios  [main, macd, rsi]
    row_heights: list       = field(default_factory=lambda: [0.55, 0.25, 0.20])

    # Colour palette — dark terminal theme
    bg_color: str           = "#0d1117"
    panel_color: str        = "#161b22"
    grid_color: str         = "#21262d"
    text_color: str         = "#c9d1d9"
    border_color: str       = "#30363d"

    # Candles
    candle_up: str          = "#3fb950"    # GitHub green
    candle_down: str        = "#f85149"    # GitHub red

    # EMA lines (SKILL.md: EMA 50 short-term, EMA 200 long-term)
    ema_short_color: str    = "#58a6ff"    # blue
    ema_long_color: str     = "#d29922"    # amber

    # MACD
    macd_line_color: str    = "#58a6ff"
    macd_signal_color: str  = "#f0883e"
    macd_hist_pos: str      = "#3fb950"
    macd_hist_neg: str      = "#f85149"

    # RSI
    rsi_line_color: str     = "#bc8cff"    # purple
    rsi_level_35: str       = "#f85149"    # sell zone lower bound
    rsi_level_50: str       = "#8b949e"    # neutral
    rsi_level_65: str       = "#3fb950"    # buy zone upper bound
    rsi_fill_buy: str       = "rgba(63,185,80,0.07)"
    rsi_fill_sell: str      = "rgba(248,81,73,0.07)"

    # Signal markers (AstroGoldHH standard — SKILL.md §Annotation)
    buy_marker_color: str   = "#3fb950"
    buy_marker_symbol: str  = "triangle-up"
    buy_marker_size: int    = 14

    sell_marker_color: str  = "#f85149"
    sell_marker_symbol: str = "triangle-down"
    sell_marker_size: int   = 14

    @property
    def html_path(self) -> Path:
        return self.output_dir / self.html_filename


CHART_CFG = ChartConfig()


# ── Individual trace builders ─────────────────────────────────────────────────

def _build_candlestick(df: pd.DataFrame, cfg: ChartConfig) -> go.Candlestick:
    """Build the main OHLC candlestick trace."""
    return go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="GC=F",
        increasing_line_color=cfg.candle_up,
        decreasing_line_color=cfg.candle_down,
        increasing_fillcolor=cfg.candle_up,
        decreasing_fillcolor=cfg.candle_down,
        line=dict(width=1),
        showlegend=True,
    )


def _build_ema_traces(
    df: pd.DataFrame, cfg: ChartConfig
) -> tuple[go.Scatter, go.Scatter]:
    """Build EMA 50 and EMA 200 line traces for the main chart."""
    ema_s_col = f"EMA_{STRATEGY_CONFIG.ema_short}"
    ema_l_col = f"EMA_{STRATEGY_CONFIG.ema_long}"

    ema_short = go.Scatter(
        x=df.index,
        y=df[ema_s_col],
        name=f"EMA {STRATEGY_CONFIG.ema_short}",
        line=dict(color=cfg.ema_short_color, width=1.5, dash="solid"),
        opacity=0.85,
    )
    ema_long = go.Scatter(
        x=df.index,
        y=df[ema_l_col],
        name=f"EMA {STRATEGY_CONFIG.ema_long}",
        line=dict(color=cfg.ema_long_color, width=1.5, dash="solid"),
        opacity=0.85,
    )
    return ema_short, ema_long


def _build_signal_traces(
    df: pd.DataFrame, cfg: ChartConfig
) -> tuple[go.Scatter, go.Scatter]:
    """
    Build BUY (▲) and SELL (▼) marker traces.
    Markers are plotted 0.5% below candle Low (BUY) and above candle High (SELL)
    so they don't overlap the candle body.
    """
    buys  = df[df["Signal"] == SIGNAL_BUY]
    sells = df[df["Signal"] == SIGNAL_SELL]

    buy_trace = go.Scatter(
        x=buys.index,
        y=buys["Low"] * 0.995,
        name="BUY Signal",
        mode="markers",
        marker=dict(
            symbol=cfg.buy_marker_symbol,
            color=cfg.buy_marker_color,
            size=cfg.buy_marker_size,
            line=dict(color="#ffffff", width=0.5),
        ),
        hovertemplate=(
            "<b>BUY</b><br>"
            "Date: %{x}<br>"
            "Close: %{customdata[0]:.2f}<br>"
            f"RSI: %{{customdata[1]:.1f}}<br>"
            f"MACD Hist: %{{customdata[2]:.4f}}<extra></extra>"
        ),
        customdata=buys[["Close", f"RSI_{STRATEGY_CONFIG.rsi_period}", "MACD_hist"]].values,
    )

    sell_trace = go.Scatter(
        x=sells.index,
        y=sells["High"] * 1.005,
        name="SELL Signal",
        mode="markers",
        marker=dict(
            symbol=cfg.sell_marker_symbol,
            color=cfg.sell_marker_color,
            size=cfg.sell_marker_size,
            line=dict(color="#ffffff", width=0.5),
        ),
        hovertemplate=(
            "<b>SELL</b><br>"
            "Date: %{x}<br>"
            "Close: %{customdata[0]:.2f}<br>"
            f"RSI: %{{customdata[1]:.1f}}<br>"
            f"MACD Hist: %{{customdata[2]:.4f}}<extra></extra>"
        ),
        customdata=sells[["Close", f"RSI_{STRATEGY_CONFIG.rsi_period}", "MACD_hist"]].values,
    )

    return buy_trace, sell_trace


def _build_macd_traces(
    df: pd.DataFrame, cfg: ChartConfig
) -> tuple[go.Bar, go.Scatter, go.Scatter]:
    """Build MACD histogram (bar) + MACD line + Signal line traces."""
    hist = df["MACD_hist"]
    hist_colors = [
        cfg.macd_hist_pos if v >= 0 else cfg.macd_hist_neg
        for v in hist
    ]

    histogram = go.Bar(
        x=df.index,
        y=hist,
        name="MACD Hist",
        marker_color=hist_colors,
        opacity=0.7,
        showlegend=True,
    )
    macd_line = go.Scatter(
        x=df.index,
        y=df["MACD_line"],
        name="MACD",
        line=dict(color=cfg.macd_line_color, width=1.2),
    )
    signal_line = go.Scatter(
        x=df.index,
        y=df["MACD_signal_line"],
        name="Signal",
        line=dict(color=cfg.macd_signal_color, width=1.2, dash="dot"),
    )
    return histogram, macd_line, signal_line


def _build_rsi_trace(df: pd.DataFrame, cfg: ChartConfig) -> go.Scatter:
    """Build RSI line trace."""
    rsi_col = f"RSI_{STRATEGY_CONFIG.rsi_period}"
    return go.Scatter(
        x=df.index,
        y=df[rsi_col],
        name=f"RSI ({STRATEGY_CONFIG.rsi_period})",
        line=dict(color=cfg.rsi_line_color, width=1.5),
    )


# ── Layout helpers ────────────────────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure, cfg: ChartConfig) -> go.Figure:
    """Apply the AstroGoldHH dark terminal theme to the full figure layout."""
    fig.update_layout(
        title=dict(
            text=cfg.title,
            font=dict(size=18, color=cfg.text_color, family="monospace"),
            x=0.01,
        ),
        height=cfg.chart_height,
        paper_bgcolor=cfg.bg_color,
        plot_bgcolor=cfg.panel_color,
        font=dict(color=cfg.text_color, family="monospace", size=11),
        legend=dict(
            bgcolor=cfg.panel_color,
            bordercolor=cfg.border_color,
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=cfg.panel_color,
            bordercolor=cfg.border_color,
            font=dict(color=cfg.text_color, size=11),
        ),
        margin=dict(l=60, r=40, t=80, b=40),
        xaxis_rangeslider_visible=False,   # disable rangeslider on main — cleaner
    )

    # Apply grid styling to all axes
    axis_style = dict(
        gridcolor=cfg.grid_color,
        gridwidth=0.5,
        zerolinecolor=cfg.grid_color,
        linecolor=cfg.border_color,
        tickfont=dict(color=cfg.text_color, size=10),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    return fig


def _add_rsi_levels(
    fig: go.Figure, cfg: ChartConfig, row: int, col: int
) -> go.Figure:
    """
    Add horizontal RSI reference lines at 35, 50, 65 (SKILL.md §Visualization).
    Uses add_hline for clean cross-plot lines.
    """
    levels = [
        (STRATEGY_CONFIG.rsi_sell_low,  cfg.rsi_level_35, "RSI 35 — Sell Zone"),
        (50.0,                           cfg.rsi_level_50, "RSI 50 — Neutral"),
        (STRATEGY_CONFIG.rsi_buy_high,  cfg.rsi_level_65, "RSI 65 — Buy Zone"),
    ]
    for level_val, color, label in levels:
        fig.add_hline(
            y=level_val,
            line=dict(color=color, width=1, dash="dash"),
            annotation_text=str(int(level_val)),
            annotation_font=dict(color=color, size=9),
            annotation_position="left",
            row=row,
            col=col,
        )

    # RSI buy zone fill (50–65)
    fig.add_hrect(
        y0=STRATEGY_CONFIG.rsi_buy_low,
        y1=STRATEGY_CONFIG.rsi_buy_high,
        fillcolor=cfg.rsi_fill_buy,
        line_width=0,
        row=row, col=col,
    )
    # RSI sell zone fill (35–50)
    fig.add_hrect(
        y0=STRATEGY_CONFIG.rsi_sell_low,
        y1=STRATEGY_CONFIG.rsi_sell_high,
        fillcolor=cfg.rsi_fill_sell,
        line_width=0,
        row=row, col=col,
    )
    return fig


def _add_signal_annotations(
    fig: go.Figure, df: pd.DataFrame, cfg: ChartConfig
) -> go.Figure:
    """
    Add lightweight text labels above/below each signal marker on the main chart
    so they remain readable when zoomed out.
    """
    buys  = df[df["Signal"] == SIGNAL_BUY]
    sells = df[df["Signal"] == SIGNAL_SELL]

    for ts, row in buys.iterrows():
        fig.add_annotation(
            x=ts, y=row["Low"] * 0.991,
            text="B", showarrow=False,
            font=dict(color=cfg.buy_marker_color, size=8, family="monospace"),
            row=1, col=1,
        )
    for ts, row in sells.iterrows():
        fig.add_annotation(
            x=ts, y=row["High"] * 1.009,
            text="S", showarrow=False,
            font=dict(color=cfg.sell_marker_color, size=8, family="monospace"),
            row=1, col=1,
        )
    return fig


# ── Main chart builder ────────────────────────────────────────────────────────

def build_chart(
    df: pd.DataFrame,
    ticker: str = "GC=F",
    cfg: ChartConfig = CHART_CFG,
) -> go.Figure:
    """
    Assemble the full AstroGoldHH dashboard figure.

    Layout (3 rows, shared x-axis):
        Row 1 (55%) — Candlestick + EMA 50/200 + BUY/SELL markers
        Row 2 (25%) — MACD histogram + MACD line + Signal line
        Row 3 (20%) — RSI (14) + levels at 35 / 50 / 65

    Parameters
    ----------
    df     : Signal-annotated DataFrame from strategy_engine.run_strategy().
    ticker : Display label (used in subplot y-axis titles).
    cfg    : ChartConfig styling overrides.

    Returns
    -------
    go.Figure — ready to render offline.
    """
    try:
        # ── Validate required columns ─────────────────────────────────────
        required = [
            "Open", "High", "Low", "Close",
            f"EMA_{STRATEGY_CONFIG.ema_short}",
            f"EMA_{STRATEGY_CONFIG.ema_long}",
            "MACD_line", "MACD_signal_line", "MACD_hist",
            f"RSI_{STRATEGY_CONFIG.rsi_period}",
            "Signal",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"build_chart: DataFrame is missing columns: {missing}. "
                "Run strategy_engine.run_strategy() before calling build_chart()."
            )

        log.info("Building AstroGoldHH dashboard for %s (%d bars) …", ticker, len(df))

        # ── Create subplot grid ───────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=cfg.row_heights,
            subplot_titles=("", "MACD (12, 26, 9)", "RSI (14)"),
        )

        # ── Row 1: Main chart ─────────────────────────────────────────────
        fig.add_trace(_build_candlestick(df, cfg),       row=1, col=1)
        ema_s, ema_l = _build_ema_traces(df, cfg)
        fig.add_trace(ema_s,                             row=1, col=1)
        fig.add_trace(ema_l,                             row=1, col=1)
        buy_m, sell_m = _build_signal_traces(df, cfg)
        fig.add_trace(buy_m,                             row=1, col=1)
        fig.add_trace(sell_m,                            row=1, col=1)

        # ── Row 2: MACD ───────────────────────────────────────────────────
        hist_t, macd_t, sig_t = _build_macd_traces(df, cfg)
        fig.add_trace(hist_t,                            row=2, col=1)
        fig.add_trace(macd_t,                            row=2, col=1)
        fig.add_trace(sig_t,                             row=2, col=1)

        # MACD zero line
        fig.add_hline(y=0, line=dict(color=cfg.grid_color, width=1), row=2, col=1)

        # ── Row 3: RSI ────────────────────────────────────────────────────
        fig.add_trace(_build_rsi_trace(df, cfg),         row=3, col=1)
        fig = _add_rsi_levels(fig, cfg, row=3, col=1)

        # ── Signal text labels on main chart ─────────────────────────────
        fig = _add_signal_annotations(fig, df, cfg)

        # ── Axis labels ───────────────────────────────────────────────────
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="MACD",        row=2, col=1)
        fig.update_yaxes(title_text="RSI",         row=3, col=1,
                         range=[0, 100], fixedrange=False)

        # ── Dark theme ────────────────────────────────────────────────────
        fig = _apply_dark_theme(fig, cfg)

        # Subtitle: subplot title colour fix
        for ann in fig.layout.annotations:
            ann.font.color = cfg.text_color
            ann.font.size  = 11

        buy_count  = (df["Signal"] == SIGNAL_BUY).sum()
        sell_count = (df["Signal"] == SIGNAL_SELL).sum()
        log.info("Chart assembled — BUY markers: %d  SELL markers: %d",
                 buy_count, sell_count)
        return fig

    except Exception as exc:
        log.error("build_chart failed: %s", exc)
        raise


# ── Render / save ─────────────────────────────────────────────────────────────

def render_offline(
    fig: go.Figure,
    cfg: ChartConfig = CHART_CFG,
    auto_open: bool = True,
) -> Path:
    """
    Save the figure as a self-contained offline HTML file (CLAUDE.md §Visualization).
    No CDN calls — the entire Plotly bundle is embedded in the HTML.

    Parameters
    ----------
    fig       : Assembled go.Figure from build_chart().
    cfg       : ChartConfig (supplies output path).
    auto_open : If True, open the HTML in the default browser after saving.

    Returns
    -------
    Path to the saved HTML file.
    """
    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg.html_path

        pyo.plot(
            fig,
            filename=str(out_path),
            auto_open=auto_open,
            include_plotlyjs=True,     # fully offline — embeds ~3 MB JS bundle
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
            },
        )

        log.info("Dashboard saved (offline) → %s", out_path)
        return out_path

    except Exception as exc:
        log.error("render_offline failed: %s", exc)
        raise


# ── Convenience pipeline ──────────────────────────────────────────────────────

def render_from_dataframe(
    df: pd.DataFrame,
    ticker: str = "GC=F",
    cfg: ChartConfig = CHART_CFG,
    auto_open: bool = True,
) -> Path:
    """
    One-call entry point: enriched DataFrame → HTML dashboard.
    Wraps build_chart() + render_offline().
    """
    try:
        fig = build_chart(df, ticker=ticker, cfg=cfg)
        return render_offline(fig, cfg=cfg, auto_open=auto_open)
    except Exception as exc:
        log.error("render_from_dataframe failed: %s", exc)
        raise


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TICKER = "GC=F"

    # Try to load pre-computed strategy output first (fastest path)
    strategy_csv = DATA_DIR / "GC_F_strategy_output.csv"

    if strategy_csv.exists():
        log.info("Loading pre-computed strategy output from %s …", strategy_csv.name)
        try:
            df = pd.read_csv(strategy_csv, index_col="Date", parse_dates=True)
            log.info("Loaded %d rows from cache.", len(df))
        except Exception as exc:
            log.warning("Failed to load cache (%s). Re-running strategy …", exc)
            df = None
    else:
        df = None

    # Fallback: load raw CSV and run strategy pipeline
    if df is None:
        try:
            from data_fetcher import load_csv, DATA_DIR, DEFAULT_INTERVAL
            raw_df = load_csv(TICKER, interval=DEFAULT_INTERVAL, data_dir=DATA_DIR)
            df = run_strategy(raw_df)
        except FileNotFoundError:
            log.error(
                "No data found for %s.\n"
                "Run:  python data_fetcher.py  →  python strategy_engine.py  first.",
                TICKER,
            )
            sys.exit(1)

    # Render the dashboard
    out_path = render_from_dataframe(df, ticker=TICKER, auto_open=True)
    print(f"\n✓ Dashboard ready → {out_path}")
