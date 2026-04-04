# Phase 1: Data & Backtesting Plumbing

## Quick Start

```bash
pip install -r requirements.txt
python run_phase1.py
```

Or via CLI with custom parameters:

```bash
python phase1_engine.py --ticker AAPL --years 3 --sma-fast 10 --sma-slow 30 --save-chart aapl_chart.png
```

## Architecture

```
phase1_engine.py
├── DataEngine          — Downloads OHLCV data, appends indicators
├── SignalGenerator      — SMA crossover signals with +1 day shift
├── Backtester           — vectorbt portfolio simulation
├── TearSheet            — Performance metrics dataclass
└── run_phase1()         — Orchestrator function
```

## Look-Ahead Bias Prevention (Critical)

The pipeline enforces a strict temporal separation:

```
Day t (Close)          Day t+1 (Open)
─────────────          ──────────────
SMA crossover  ──────► Trade executed
detected here          here

Signal is SHIFTED +1 day before being passed to vectorbt.
Execution uses the OPEN price (not Close) of the signal day.
```

This means:
1. Crossover detected using closing prices on day `t`
2. `SignalGenerator.generate()` shifts the boolean signal forward by 1 day
3. `Backtester.run()` executes at the **Open** price of day `t+1`
4. No future information is used for any decision

## Indicators

| Indicator | Library    | Look-Ahead Safe? | Notes                        |
|-----------|------------|-------------------|------------------------------|
| SMA(20)   | pandas-ta  | Yes               | Trailing 20-day mean         |
| SMA(50)   | pandas-ta  | Yes               | Trailing 50-day mean         |
| RSI(14)   | pandas-ta  | Yes               | Trailing 14-day momentum     |

## Tear Sheet Metrics

- **Total Return %** — Strategy cumulative return
- **Buy & Hold Return %** — Benchmark passive return
- **Annualized Return %** — CAGR
- **Annualized Volatility %** — Std dev of daily returns × √252
- **Sharpe Ratio** — Risk-adjusted return (from vectorbt)
- **Max Drawdown %** — Largest peak-to-trough decline
- **Total Trades** — Number of round-trip trades
- **Win Rate %** — Percentage of profitable trades

## Next Phases

- **Phase 2**: FinBERT sentiment pipeline
- **Phase 3**: ML signal model with time-series CV
- **Phase 4**: Full backtest engine combining technical + sentiment
- **Phase 5**: Orchestration and reporting
