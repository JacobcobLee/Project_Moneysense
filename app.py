"""
Phase 1.5 — Task 2: Streamlit Trading Dashboard
=================================================
Consumes the persistence layer from Task 1:
  - daily_results.csv       → time-series charts + signal table
  - performance_metrics.json → KPI metric cards + header timestamp

Run locally:
    streamlit run app.py

Layout:
    ┌─────────────────────────────────────────────────┐
    │  Title  ·  Last Refreshed: 2025-06-15T20:15:00Z │
    ├──────┬──────┬──────┬──────┬──────┬──────────────┤
    │ Tot  │ B&H  │ Ann  │Sharpe│ Max  │  Win         │
    │ Ret  │ Ret  │ Ret  │Ratio │ DD   │  Rate        │
    ├─────────────────────────────────────────────────┤
    │  Equity Curve (strategy vs buy & hold)          │
    ├─────────────────────────────────────────────────┤
    │  Price Chart (Close + SMAs + entry/exit marks)  │
    ├─────────────────────────────────────────────────┤
    │  RSI Subplot                                     │
    ├─────────────────────────────────────────────────┤
    │  Recent Signals Table (last 5 days)             │
    └─────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
RESULTS_CSV = Path("daily_results.csv")
METRICS_JSON = Path("performance_metrics.json")

# Streamlit page config — MUST be the first st.* call
st.set_page_config(
    page_title="SMA Crossover Dashboard",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING (cached to avoid re-reads on every interaction)
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # Re-read files every 5 minutes
def load_metrics() -> dict:
    """Load scalar performance metrics from JSON."""
    if not METRICS_JSON.exists():
        return {}
    with open(METRICS_JSON, "r") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    """Load time-series signal data from CSV."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV, parse_dates=["Date"], index_col="Date")
    return df


# ─────────────────────────────────────────────────────────────────────
# HELPER: format the "Last Refreshed" timestamp
# ─────────────────────────────────────────────────────────────────────

def format_timestamp(iso_str: str) -> str:
    """
    Convert UTC ISO timestamp to a human-readable string with
    relative time (e.g., '2 hours ago').
    """
    try:
        run_time = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(run_time.tzinfo)
        delta = now - run_time
        hours = delta.total_seconds() / 3600

        if hours < 1:
            relative = f"{int(delta.total_seconds() / 60)} minutes ago"
        elif hours < 24:
            relative = f"{int(hours)} hours ago"
        else:
            relative = f"{int(hours / 24)} days ago"

        formatted = run_time.strftime("%Y-%m-%d %H:%M UTC")
        return f"{formatted}  ({relative})"
    except (ValueError, TypeError):
        return iso_str or "Unknown"


# ─────────────────────────────────────────────────────────────────────
# HELPER: build Plotly charts
# ─────────────────────────────────────────────────────────────────────

def build_equity_chart(df: pd.DataFrame, init_cash: float = 100_000) -> go.Figure:
    """
    Equity curve: strategy portfolio value over time.
    Includes a buy-and-hold reference line for comparison.
    """
    # Compute buy-and-hold equity from Close prices
    bh_equity = (df["Close"] / df["Close"].iloc[0]) * init_cash

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity"],
        name="SMA Crossover Strategy",
        line=dict(color="#2962FF", width=2.5),
        hovertemplate="Strategy: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=bh_equity,
        name="Buy & Hold",
        line=dict(color="#78909C", width=1.5, dash="dot"),
        hovertemplate="Buy & Hold: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Portfolio Equity Curve",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )

    return fig


def build_price_chart(df: pd.DataFrame) -> go.Figure:
    """
    Price chart with SMA overlays and entry/exit signal markers.

    Entry signals → green triangle-up on the price line.
    Exit signals  → red triangle-down on the price line.
    Background shading shows Long (green) vs Flat (no shade) periods.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & SMA Crossover", "RSI (14)"),
    )

    # ── Close price ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close",
        line=dict(color="#263238", width=1.8),
        hovertemplate="Close: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── SMA lines ──
    sma_fast_col = [c for c in df.columns if c.startswith("SMA_") and "20" in c]
    sma_slow_col = [c for c in df.columns if c.startswith("SMA_") and "50" in c]

    if sma_fast_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sma_fast_col[0]],
            name=sma_fast_col[0],
            line=dict(color="#FF6D00", width=1.2),
            hovertemplate=f"{sma_fast_col[0]}: $%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    if sma_slow_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[sma_slow_col[0]],
            name=sma_slow_col[0],
            line=dict(color="#AA00FF", width=1.2),
            hovertemplate=f"{sma_slow_col[0]}: $%{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

    # ── Entry/Exit markers ──
    entries = df[df["Entry_Signal"] == 1]
    exits = df[df["Exit_Signal"] == 1]

    if not entries.empty:
        fig.add_trace(go.Scatter(
            x=entries.index, y=entries["Close"],
            mode="markers",
            name="Entry (Buy)",
            marker=dict(symbol="triangle-up", size=14, color="#00C853",
                        line=dict(width=1, color="#1B5E20")),
            hovertemplate="BUY @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits.index, y=exits["Close"],
            mode="markers",
            name="Exit (Sell)",
            marker=dict(symbol="triangle-down", size=14, color="#FF1744",
                        line=dict(width=1, color="#B71C1C")),
            hovertemplate="SELL @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Position shading (long periods as light green background) ──
    # Find contiguous long periods
    position = df["Position"]
    in_long = False
    for i in range(len(df)):
        if position.iloc[i] == 1 and not in_long:
            x0 = df.index[i]
            in_long = True
        elif position.iloc[i] == 0 and in_long:
            x1 = df.index[i]
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(0, 200, 83, 0.08)",
                layer="below", line_width=0,
                row=1, col=1,
            )
            in_long = False
    # Close any open long at the end
    if in_long:
        fig.add_vrect(
            x0=x0, x1=df.index[-1],
            fillcolor="rgba(0, 200, 83, 0.08)",
            layer="below", line_width=0,
            row=1, col=1,
        )

    # ── RSI subplot ──
    rsi_col = [c for c in df.columns if c.startswith("RSI_")]
    if rsi_col:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[rsi_col[0]],
            name=rsi_col[0],
            line=dict(color="#5C6BC0", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1)

        # Overbought/oversold reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                       opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                       opacity=0.5, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(92, 107, 192, 0.05)",
                       layer="below", line_width=0, row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=550,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    return fig


# ─────────────────────────────────────────────────────────────────────
# HELPER: style the recent signals table
# ─────────────────────────────────────────────────────────────────────

def style_signal_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply conditional formatting to the signals table:
      - Entry_Signal = 1 → green background
      - Exit_Signal  = 1 → red background
      - Position = 1     → blue text (LONG), 0 → grey text (FLAT)
    """
    def highlight_signals(row):
        styles = [""] * len(row)
        col_names = list(row.index)

        if "Entry_Signal" in col_names and row["Entry_Signal"] == 1:
            idx = col_names.index("Entry_Signal")
            styles[idx] = "background-color: #C8E6C9; font-weight: bold"
        if "Exit_Signal" in col_names and row["Exit_Signal"] == 1:
            idx = col_names.index("Exit_Signal")
            styles[idx] = "background-color: #FFCDD2; font-weight: bold"
        if "Position" in col_names:
            idx = col_names.index("Position")
            if row["Position"] == 1:
                styles[idx] = "color: #1565C0; font-weight: bold"
            else:
                styles[idx] = "color: #9E9E9E"

        return styles

    return df.style.apply(highlight_signals, axis=1).format({
        "Close": "${:.2f}",
        "Open": "${:.2f}",
        "High": "${:.2f}",
        "Low": "${:.2f}",
        "Equity": "${:,.0f}",
    }, na_rep="—")


# ─────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ──
    metrics = load_metrics()
    df = load_results()

    # ── Guard: check if data files exist ──
    if not metrics or df.empty:
        st.error(
            "⚠️ **Data files not found.** Run the pipeline first:\n\n"
            "```bash\npython run_phase1.py\n```\n\n"
            "This will generate `daily_results.csv` and `performance_metrics.json`."
        )
        st.stop()

    # ══════════════════════════════════════════════════════════════
    # SECTION 1: HEADER
    # ══════════════════════════════════════════════════════════════

    col_title, col_meta = st.columns([3, 2])

    with col_title:
        st.title("📈 SMA Crossover Dashboard")
    with col_meta:
        ticker = metrics.get("ticker", "—")
        strategy = metrics.get("strategy", "—")
        period = f"{metrics.get('period_start', '?')} → {metrics.get('period_end', '?')}"
        refreshed = format_timestamp(metrics.get("run_timestamp_utc", ""))

        st.markdown(
            f"**{ticker}** · {strategy}  \n"
            f"📅 {period}  \n"
            f"🔄 Last Refreshed: `{refreshed}`"
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # SECTION 2: KPI METRIC CARDS
    # ══════════════════════════════════════════════════════════════

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total_ret = metrics.get("total_return_pct", 0)
    bh_ret = metrics.get("buy_and_hold_return_pct", 0)

    k1.metric(
        label="Total Return",
        value=f"{total_ret:+.2f}%",
        delta=f"{total_ret - bh_ret:+.2f}% vs B&H",
        delta_color="normal",
    )
    k2.metric(
        label="Buy & Hold",
        value=f"{bh_ret:+.2f}%",
    )
    k3.metric(
        label="Annualized Return",
        value=f"{metrics.get('annualized_return_pct', 0):+.2f}%",
    )
    k4.metric(
        label="Sharpe Ratio",
        value=f"{metrics.get('sharpe_ratio', 0):.3f}",
    )
    k5.metric(
        label="Max Drawdown",
        value=f"{metrics.get('max_drawdown_pct', 0):.2f}%",
    )
    k6.metric(
        label="Win Rate",
        value=f"{metrics.get('win_rate_pct', 0):.1f}%",
        delta=f"{metrics.get('total_trades', 0)} trades",
        delta_color="off",
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # SECTION 3: CHARTS
    # ══════════════════════════════════════════════════════════════

    # Equity curve
    equity_fig = build_equity_chart(df)
    st.plotly_chart(equity_fig, use_container_width=True)

    # Price + SMA + signals + RSI
    price_fig = build_price_chart(df)
    st.plotly_chart(price_fig, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # SECTION 4: RECENT SIGNALS TABLE (last 5 days)
    # ══════════════════════════════════════════════════════════════

    st.subheader("🔔 Recent Signals (Last 5 Trading Days)")

    # Select the columns most relevant for the operator
    display_cols = [
        "Open", "Close",
        "Entry_Signal", "Exit_Signal", "Position", "Equity",
    ]
    # Add SMA/RSI columns if present
    for col in df.columns:
        if col.startswith("SMA_") or col.startswith("RSI_"):
            display_cols.insert(2, col)

    # De-duplicate while preserving order
    display_cols = list(dict.fromkeys(display_cols))
    # Filter to columns that actually exist
    display_cols = [c for c in display_cols if c in df.columns]

    recent = df[display_cols].tail(5).copy()
    recent.index = recent.index.strftime("%Y-%m-%d")  # Clean date format

    # ── Current position callout ──
    current_pos = df["Position"].iloc[-1]
    pos_label = "🟢 **LONG**" if current_pos == 1 else "⚪ **FLAT**"
    st.markdown(f"Current Position: {pos_label}")

    # ── Styled table ──
    styled = style_signal_table(recent)
    st.dataframe(styled, use_container_width=True, height=220)

    # ── Footer ──
    st.divider()
    st.caption(
        "⚠️ This is a research backtest dashboard — not live trading advice. "
        "Past performance does not indicate future results. "
        "All signals are shifted +1 day to prevent look-ahead bias."
    )


if __name__ == "__main__":
    main()
