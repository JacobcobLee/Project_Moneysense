"""
Phase 1: Data & Backtesting Plumbing
=====================================
Modular engine for:
  1. Historical daily data ingestion (yfinance)
  2. Technical indicator computation (pandas-ta)
  3. SMA crossover baseline strategy (vectorbt)
  4. Performance tear sheet generation

GUARDRAILS ENFORCED:
  - Look-Ahead Bias: All signals are shifted forward by 1 day.
    A crossover detected at day-t close generates an entry/exit on day t+1.
  - No future data leakage in indicator computation (pandas-ta uses
    only trailing windows by construction).

Author: Quant Research Team
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────
# 1. DATA ENGINE
# ─────────────────────────────────────────────────────────────────────

class DataEngine:
    """
    Downloads OHLCV data and appends technical indicators.

    All indicators use ONLY historical (trailing) data by construction:
      - SMA(20) on day t uses closes from t-19 to t.
      - RSI(14) on day t uses the last 14 price changes ending at t.

    No future data is referenced.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        years: int = 5,
        end_date: Optional[str] = None,
    ):
        self.ticker = ticker.upper()
        self.years = years
        self.end_date = (
            pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
        )
        self.start_date = self.end_date - pd.DateOffset(years=years)
        self.raw_data: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None

    def download(self) -> pd.DataFrame:
        """Download daily OHLCV data from Yahoo Finance."""
        print(
            f"[DataEngine] Downloading {self.ticker} "
            f"from {self.start_date.date()} to {self.end_date.date()}..."
        )
        df = yf.download(
            self.ticker,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,  # Use adjusted prices to handle splits/dividends
        )

        if df.empty:
            raise ValueError(
                f"No data returned for {self.ticker}. Check ticker/date range."
            )

        # Flatten multi-level columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure clean column names
        df.columns = [c.strip().title() for c in df.columns]

        # Validate required columns exist
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in downloaded data: {missing}")

        self.raw_data = df.copy()
        self.data = df.copy()
        print(f"[DataEngine] Downloaded {len(df)} trading days.")
        return self.data

    def add_indicators(
        self,
        sma_fast: int = 20,
        sma_slow: int = 50,
        rsi_period: int = 14,
    ) -> pd.DataFrame:
        """
        Append SMA and RSI indicators.

        These are TRAILING indicators by construction:
          - SMA(n) at index t = mean(Close[t-n+1 : t+1])
          - RSI(n) at index t uses the last n price changes up to t

        The first `sma_slow - 1` rows will have NaN for SMA_slow.
        We keep NaNs here; they will be handled at signal generation time.
        """
        if self.data is None:
            raise RuntimeError("Call .download() before .add_indicators().")

        df = self.data

        # ── SMA ──
        df[f"SMA_{sma_fast}"] = ta.sma(df["Close"], length=sma_fast)
        df[f"SMA_{sma_slow}"] = ta.sma(df["Close"], length=sma_slow)

        # ── RSI ──
        df[f"RSI_{rsi_period}"] = ta.rsi(df["Close"], length=rsi_period)

        self.data = df
        self._sma_fast_col = f"SMA_{sma_fast}"
        self._sma_slow_col = f"SMA_{sma_slow}"
        self._rsi_col = f"RSI_{rsi_period}"

        nan_rows = df[[self._sma_fast_col, self._sma_slow_col, self._rsi_col]].isna().any(axis=1).sum()
        print(
            f"[DataEngine] Added SMA({sma_fast}), SMA({sma_slow}), RSI({rsi_period}). "
            f"Warm-up NaN rows: {nan_rows}"
        )
        return self.data

    def get_clean_data(self) -> pd.DataFrame:
        """Return data with warm-up NaN rows dropped."""
        if self.data is None:
            raise RuntimeError("No data available. Run download + add_indicators first.")
        return self.data.dropna().copy()


# ─────────────────────────────────────────────────────────────────────
# 2. SIGNAL GENERATOR (with look-ahead bias prevention)
# ─────────────────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Generates entry/exit signals from SMA crossover.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  CRITICAL: LOOK-AHEAD BIAS PREVENTION                         ║
    ║                                                                ║
    ║  A crossover is detected using day-t CLOSE prices.            ║
    ║  Therefore, the signal is SHIFTED FORWARD by 1 day:           ║
    ║    - Crossover detected at close of day t                     ║
    ║    - Trade is executed at OPEN of day t+1                     ║
    ║                                                                ║
    ║  In vectorbt, we pass shifted signals so the portfolio        ║
    ║  enters/exits one bar AFTER the crossover is observed.        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, data: pd.DataFrame, sma_fast_col: str, sma_slow_col: str):
        self.data = data.copy()
        self.sma_fast_col = sma_fast_col
        self.sma_slow_col = sma_slow_col
        self.entries: Optional[pd.Series] = None
        self.exits: Optional[pd.Series] = None

    def generate(self) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry/exit boolean signals with 1-day forward shift.

        Entry: SMA_fast crosses ABOVE SMA_slow (golden cross)
        Exit:  SMA_fast crosses BELOW SMA_slow (death cross)

        Returns shifted signals aligned to the NEXT trading day.
        """
        fast = self.data[self.sma_fast_col]
        slow = self.data[self.sma_slow_col]

        # ── Raw crossover detection (computed at day-t close) ──
        # Golden cross: fast was below slow yesterday, fast >= slow today
        raw_entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        # Death cross: fast was above slow yesterday, fast <= slow today
        raw_exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        # ══════════════════════════════════════════════════════
        # SHIFT FORWARD BY 1 DAY to prevent look-ahead bias.
        # Signal on day t → execution on day t+1.
        # ══════════════════════════════════════════════════════
        self.entries = raw_entries.shift(1).fillna(False).astype(bool)
        self.exits = raw_exits.shift(1).fillna(False).astype(bool)

        n_entries = self.entries.sum()
        n_exits = self.exits.sum()
        print(
            f"[SignalGenerator] Generated {n_entries} entry signals, "
            f"{n_exits} exit signals (shifted +1 day)."
        )

        return self.entries, self.exits


# ─────────────────────────────────────────────────────────────────────
# 3. BACKTESTER
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TearSheet:
    """Performance summary for a backtest run."""

    ticker: str
    strategy: str
    period_start: str
    period_end: str
    total_return_pct: float
    buy_and_hold_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    sharpe_ratio: float
    annualized_return_pct: float
    annualized_volatility_pct: float

    def display(self) -> None:
        """Print a formatted tear sheet to console."""
        divider = "=" * 56
        print(f"\n{divider}")
        print(f"  PERFORMANCE TEAR SHEET")
        print(f"{divider}")
        print(f"  Ticker:              {self.ticker}")
        print(f"  Strategy:            {self.strategy}")
        print(f"  Period:              {self.period_start} → {self.period_end}")
        print(f"{'-' * 56}")
        print(f"  Total Return:        {self.total_return_pct:>+10.2f}%")
        print(f"  Buy & Hold Return:   {self.buy_and_hold_return_pct:>+10.2f}%")
        print(f"  Annualized Return:   {self.annualized_return_pct:>+10.2f}%")
        print(f"  Annualized Vol:      {self.annualized_volatility_pct:>10.2f}%")
        print(f"  Sharpe Ratio:        {self.sharpe_ratio:>10.3f}")
        print(f"  Max Drawdown:        {self.max_drawdown_pct:>10.2f}%")
        print(f"{'-' * 56}")
        print(f"  Total Trades:        {self.total_trades:>10d}")
        print(f"  Win Rate:            {self.win_rate_pct:>10.2f}%")
        print(f"{divider}\n")


class Backtester:
    """
    Runs a vectorbt portfolio simulation and produces a tear sheet.

    Execution model:
      - Signals are ALREADY shifted +1 day by SignalGenerator.
      - Portfolio enters at the OPEN price on the signal day.
      - This means: crossover at close(t) → enter at open(t+1).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entries: pd.Series,
        exits: pd.Series,
        ticker: str = "SPY",
        init_cash: float = 100_000.0,
        fees: float = 0.001,  # 10 bps per trade
    ):
        self.data = data
        self.entries = entries
        self.exits = exits
        self.ticker = ticker
        self.init_cash = init_cash
        self.fees = fees
        self.portfolio: Optional[vbt.Portfolio] = None

    def run(self) -> vbt.Portfolio:
        """
        Execute the backtest using vectorbt.

        We use the 'Open' price for execution to model realistic fills:
        signal observed at close → filled at next day's open.
        Since signals are already shifted, we execute at the Open of
        the signal day (which IS the next day's open relative to detection).

        NOTE: vectorbt API varies across versions. We use a compatibility
        wrapper that tries the modern API first, then falls back to the
        minimal API that works across all 0.2x versions.
        """
        # Align entries/exits to data index (safety check)
        entries = self.entries.reindex(self.data.index).fillna(False).astype(bool)
        exits = self.exits.reindex(self.data.index).fillna(False).astype(bool)

        # ── Base kwargs (work in ALL vectorbt 0.2x versions) ──
        base_kwargs = dict(
            close=self.data["Close"],
            open=self.data["Open"],
            entries=entries,
            exits=exits,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=0.0005,       # 5 bps slippage model
            freq="1D",
        )

        # ── Extended kwargs (only in some versions) ──
        # Try adding them; if from_signals rejects any, fall back.
        extended_kwargs = dict(
            **base_kwargs,
            upon_opposite_signal="close",
        )

        try:
            self.portfolio = vbt.Portfolio.from_signals(**extended_kwargs)
        except TypeError:
            # Older/newer vectorbt without upon_opposite_signal.
            # Default behavior (accumulate=False) already closes on
            # opposite signal, so the base call is functionally identical.
            self.portfolio = vbt.Portfolio.from_signals(**base_kwargs)

        print(f"[Backtester] Simulation complete. {self.portfolio.trades.count()} trades executed.")
        return self.portfolio

    def tear_sheet(self) -> TearSheet:
        """Build and return a TearSheet dataclass from portfolio stats."""
        if self.portfolio is None:
            raise RuntimeError("Call .run() before .tear_sheet().")

        pf = self.portfolio
        stats = pf.stats()

        # Compute buy-and-hold benchmark
        bh_return = (
            (self.data["Close"].iloc[-1] / self.data["Close"].iloc[0]) - 1
        ) * 100

        # Extract metrics safely
        total_return = float(stats.get("Total Return [%]", 0.0))
        max_dd = float(stats.get("Max Drawdown [%]", 0.0))
        total_trades = int(stats.get("Total Trades", 0))
        win_rate = float(stats.get("Win Rate [%]", 0.0))
        sharpe = float(stats.get("Sharpe Ratio", 0.0))

        # Compute annualized metrics
        n_days = (self.data.index[-1] - self.data.index[0]).days
        n_years = n_days / 365.25
        ann_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        daily_returns = pf.returns()
        ann_vol = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 1 else 0

        ts = TearSheet(
            ticker=self.ticker,
            strategy="SMA Crossover (20/50)",
            period_start=str(self.data.index[0].date()),
            period_end=str(self.data.index[-1].date()),
            total_return_pct=total_return,
            buy_and_hold_return_pct=round(bh_return, 2),
            max_drawdown_pct=max_dd,
            total_trades=total_trades,
            win_rate_pct=win_rate,
            sharpe_ratio=round(sharpe, 3),
            annualized_return_pct=round(ann_return, 2),
            annualized_volatility_pct=round(ann_vol, 2),
        )

        return ts

    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """Plot portfolio equity curve vs buy-and-hold."""
        if self.portfolio is None:
            raise RuntimeError("Call .run() before .plot_equity_curve().")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # ── Equity curve ──
        equity = self.portfolio.value()
        bh_equity = (
            self.data["Close"] / self.data["Close"].iloc[0]
        ) * self.init_cash

        axes[0].plot(equity.index, equity.values, label="SMA Crossover Strategy", linewidth=1.5)
        axes[0].plot(bh_equity.index, bh_equity.values, label="Buy & Hold", linewidth=1.0, alpha=0.7)
        axes[0].set_title(f"{self.ticker} — SMA Crossover vs Buy & Hold", fontsize=14)
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # ── Drawdown ──
        drawdown = self.portfolio.drawdown() * 100
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
        axes[1].set_title("Drawdown (%)", fontsize=12)
        axes[1].set_ylabel("Drawdown %")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Backtester] Chart saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATOR — ties it all together
# ─────────────────────────────────────────────────────────────────────

def run_phase1(
    ticker: str = "SPY",
    years: int = 5,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
    init_cash: float = 100_000.0,
    save_chart: Optional[str] = None,
) -> TearSheet:
    """
    Full Phase 1 pipeline:
      1. Download data
      2. Compute indicators
      3. Generate signals (with look-ahead bias prevention)
      4. Run backtest
      5. Output tear sheet

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    years : int
        Number of years of historical data.
    sma_fast : int
        Fast SMA period (default 20).
    sma_slow : int
        Slow SMA period (default 50).
    rsi_period : int
        RSI lookback period (default 14).
    init_cash : float
        Starting capital for the simulation.
    save_chart : str, optional
        If provided, saves equity curve chart to this path.

    Returns
    -------
    TearSheet
        Dataclass containing all performance metrics.
    """
    print("=" * 56)
    print("  PHASE 1: DATA & BACKTESTING PLUMBING")
    print("=" * 56)

    # Step 1: Data ingestion
    engine = DataEngine(ticker=ticker, years=years)
    engine.download()

    # Step 2: Technical indicators
    engine.add_indicators(sma_fast=sma_fast, sma_slow=sma_slow, rsi_period=rsi_period)

    # Step 3: Clean data (drop warm-up NaN rows)
    clean_data = engine.get_clean_data()
    print(f"[Pipeline] Clean dataset: {len(clean_data)} rows "
          f"({clean_data.index[0].date()} to {clean_data.index[-1].date()})")

    # Step 4: Signal generation (shifted +1 day)
    sig_gen = SignalGenerator(
        data=clean_data,
        sma_fast_col=f"SMA_{sma_fast}",
        sma_slow_col=f"SMA_{sma_slow}",
    )
    entries, exits = sig_gen.generate()

    # Step 5: Backtest
    bt = Backtester(
        data=clean_data,
        entries=entries,
        exits=exits,
        ticker=ticker,
        init_cash=init_cash,
    )
    bt.run()

    # Step 6: Tear sheet
    ts = bt.tear_sheet()
    ts.display()

    # Step 7: Optional chart
    if save_chart:
        bt.plot_equity_curve(save_path=save_chart)

    return ts


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: SMA Crossover Backtest")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol")
    parser.add_argument("--years", type=int, default=5, help="Years of history")
    parser.add_argument("--sma-fast", type=int, default=20, help="Fast SMA period")
    parser.add_argument("--sma-slow", type=int, default=50, help="Slow SMA period")
    parser.add_argument("--rsi", type=int, default=14, help="RSI period")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    parser.add_argument("--save-chart", type=str, default=None, help="Path to save chart")
    args = parser.parse_args()

    result = run_phase1(
        ticker=args.ticker,
        years=args.years,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        rsi_period=args.rsi,
        init_cash=args.cash,
        save_chart=args.save_chart,
    )
