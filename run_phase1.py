"""
Phase 1.5 — Task 1: Runner with Persistence
=============================================
Executes the Phase 1 backtest pipeline and persists two output files:

  1. daily_results.csv  — Last 60 trading days of OHLCV + indicators +
                          entry/exit signals + portfolio equity.
                          This is what the Streamlit dashboard reads.

  2. performance_metrics.json — Full tear sheet as structured JSON.
                                Scalar metrics for dashboard KPI cards.

WHY TWO FILES:
  - CSV is ideal for time-series rows (dates, prices, signals).
  - JSON is ideal for scalar metrics (total return, Sharpe, etc.).
  - Mixing both into one CSV requires fragile header parsing.
  - Two files = clean contract for the Streamlit consumer in Task 2.

Usage:
    pip install -r requirements.txt
    python run_phase1.py                     # defaults: SPY, 5y
    python run_phase1.py --ticker AAPL       # custom ticker
    python run_phase1.py --ticker QQQ --years 3 --recent-days 90

Output:
    ./daily_results.csv
    ./performance_metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from phase1_engine import (
    Backtester,
    DataEngine,
    SignalGenerator,
    TearSheet,
)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(".")  # Root directory — where GH Actions will commit from
RESULTS_CSV = OUTPUT_DIR / "daily_results.csv"
METRICS_JSON = OUTPUT_DIR / "performance_metrics.json"


def build_signal_dataframe(
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    portfolio,
    recent_days: int = 60,
) -> pd.DataFrame:
    """
    Combine OHLCV + indicators + signals + equity into a single
    DataFrame for dashboard consumption.

    Parameters
    ----------
    data : pd.DataFrame
        Clean data with OHLCV + indicator columns.
    entries : pd.Series
        Boolean entry signals (already shifted +1 day).
    exits : pd.Series
        Boolean exit signals (already shifted +1 day).
    portfolio : vbt.Portfolio
        The executed portfolio (for equity curve extraction).
    recent_days : int
        Number of most recent trading days to include.

    Returns
    -------
    pd.DataFrame
        Dashboard-ready DataFrame with the most recent N days.
    """
    df = data.copy()

    # ── Attach signals ──
    df["Entry_Signal"] = entries.reindex(df.index).fillna(False).astype(int)
    df["Exit_Signal"] = exits.reindex(df.index).fillna(False).astype(int)

    # ── Derive a human-readable position column ──
    # 1 = Long, 0 = Flat. Forward-fill from entry/exit signals.
    position = pd.Series(0, index=df.index, dtype=int)
    in_position = False
    for i, idx in enumerate(df.index):
        if df.loc[idx, "Entry_Signal"] == 1:
            in_position = True
        elif df.loc[idx, "Exit_Signal"] == 1:
            in_position = False
        position.iloc[i] = 1 if in_position else 0
    df["Position"] = position

    # ── Attach portfolio equity curve ──
    equity = portfolio.value()
    df["Equity"] = equity.reindex(df.index)

    # ── Trim to recent N trading days ──
    df_recent = df.tail(recent_days).copy()

    # ── Clean index for CSV serialization ──
    df_recent.index.name = "Date"

    return df_recent


def save_metrics(tear_sheet: TearSheet, path: Path) -> None:
    """
    Serialize TearSheet to JSON with a run timestamp.

    The timestamp tells the Streamlit app when data was last refreshed.
    """
    metrics = asdict(tear_sheet)
    metrics["run_timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Persistence] Metrics saved to {path}")


def save_results(df: pd.DataFrame, path: Path) -> None:
    """Save signal DataFrame to CSV with proper formatting."""
    # Round floats for readability
    float_cols = df.select_dtypes(include=["float64"]).columns
    df_out = df.copy()
    df_out[float_cols] = df_out[float_cols].round(4)

    df_out.to_csv(path)
    print(f"[Persistence] Results saved to {path} ({len(df_out)} rows)")


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def main(
    ticker: str = "SPY",
    years: int = 5,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
    init_cash: float = 100_000.0,
    recent_days: int = 60,
) -> None:
    """
    Full pipeline: download → indicators → signals → backtest → persist.

    Uses the Phase 1 engine components directly (not the run_phase1
    orchestrator) so we retain access to intermediate objects needed
    for CSV persistence.
    """
    print("=" * 56)
    print("  PHASE 1.5: BACKTEST + PERSISTENCE PIPELINE")
    print("=" * 56)

    # ── Step 1: Data ingestion ──
    engine = DataEngine(ticker=ticker, years=years)
    engine.download()
    engine.add_indicators(
        sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period
    )
    clean_data = engine.get_clean_data()

    print(
        f"[Pipeline] Clean dataset: {len(clean_data)} rows "
        f"({clean_data.index[0].date()} → {clean_data.index[-1].date()})"
    )

    # ── Step 2: Signal generation (shifted +1 day) ──
    sig_gen = SignalGenerator(
        data=clean_data,
        sma_fast_col=f"SMA_{sma_fast}",
        sma_slow_col=f"SMA_{sma_slow}",
    )
    entries, exits = sig_gen.generate()

    # ── Step 3: Backtest ──
    bt = Backtester(
        data=clean_data,
        entries=entries,
        exits=exits,
        ticker=ticker,
        init_cash=init_cash,
    )
    bt.run()

    # ── Step 4: Tear sheet ──
    tear_sheet = bt.tear_sheet()
    tear_sheet.display()

    # ── Step 5: Build signal DataFrame for dashboard ──
    signal_df = build_signal_dataframe(
        data=clean_data,
        entries=entries,
        exits=exits,
        portfolio=bt.portfolio,
        recent_days=recent_days,
    )

    # ── Step 6: Persist outputs ──
    save_results(signal_df, RESULTS_CSV)
    save_metrics(tear_sheet, METRICS_JSON)

    # ── Step 7: Save equity chart (optional, for repo README) ──
    bt.plot_equity_curve(save_path="equity_curve.png")

    print("\n[Pipeline] Done. Files ready for Streamlit dashboard:")
    print(f"  → {RESULTS_CSV}")
    print(f"  → {METRICS_JSON}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1.5: Backtest with persistence for Streamlit"
    )
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument("--sma-fast", type=int, default=20, help="Fast SMA period")
    parser.add_argument("--sma-slow", type=int, default=50, help="Slow SMA period")
    parser.add_argument("--rsi", type=int, default=14, help="RSI period")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    parser.add_argument(
        "--recent-days", type=int, default=60,
        help="Number of recent trading days to include in CSV"
    )
    args = parser.parse_args()

    main(
        ticker=args.ticker,
        years=args.years,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        rsi_period=args.rsi,
        init_cash=args.cash,
        recent_days=args.recent_days,
    )
