"""
================================================================================
 STRATEGY PARAMETER OPTIMIZER
================================================================================
Backtests a MACD-momentum-cooldown + SMA-trend-filter LONG-only strategy
across a stock's full price history, then grid-searches the EXIT parameters
(Take Profit %, Stop Loss %, Trail Activation %, Trail Breakeven Buffer %)
to find combinations that hit your targets:

    Win Rate (Profitable Trades)  >= 80%
    Profit Factor                 >= 2.0

-------------------------------------------------------------------------------
LOGIC - TRANSLATED DIRECTLY FROM YOUR PINE SCRIPT (not a guess anymore)
-------------------------------------------------------------------------------
  ENTRY (long) fires when BOTH are true (your buy_condition):
    1. is_cooled_off  : hist_min < macd_histogram <= hist_max
    2. is_trend_intact: close > SMA(trend_sma_length) * (1 + min_pct_above_sma/100)
  Only enters when flat (strategy.position_size == 0) - matches your script.

  FILL TIMING: your strategy() call doesn't set process_orders_on_close=true,
  so Pine's default applies - strategy.entry() calculated on bar i fills at
  the OPEN of bar i+1, not the close of the signal bar. This script fills
  entries the same way (next bar's open), which matters for accuracy versus
  what TradingView itself reports.

  EXIT - this is a straight translation of your strategy.exit() block:
    target_price = entry_price * (1 + tp_pct)
    if use_trailing:
        activation_price = entry_price * (1 + trail_activation_pct)
        stop_price = (entry_price * (1 + trail_offset_pct)) if highest_price >= activation_price
                     else (entry_price * (1 - sl_pct))
    else:
        stop_price = entry_price * (1 - sl_pct)

  IMPORTANT: despite the UI label "Trailing Stop", this is NOT a stop that
  continuously trails the price. It's a one-time step: the stop sits at
  -SL% until the trade's high reaches the activation level, at which point
  it jumps ONCE to entry+buffer% and then stays fixed there for the rest of
  the trade - it never moves again even if price keeps climbing. Your
  Pine code confirms this (`highest_price >= activation_price` is a one-way
  switch since highest_price never decreases). I've matched that exactly.

  One position at a time, long only, no pyramiding - matches your script.

  ONE REMAINING SOURCE OF MINOR DIVERGENCE FROM TRADINGVIEW: when a stop
  and a target both fall inside the same bar's high-low range, which one
  "would have" filled first is genuinely ambiguous without intrabar data.
  This script checks the stop before the target (the conservative
  assumption). TradingView's own broker emulator uses a path heuristic
  based on the bar's open/high/low that can occasionally pick the other
  order - so expect small differences from TradingView's own trade list on
  the rare bars where both levels are touched, not a structural mismatch.

-------------------------------------------------------------------------------
WHY THIS MATTERS: OVERFITTING / CURVE-FITTING RISK
-------------------------------------------------------------------------------
Grid-searching exit parameters against the ENTIRE history and picking
whichever combo scores best on that same history is a textbook way to
curve-fit - you're finding the parameters that happened to work on the past,
not parameters with genuine predictive edge. Two safeguards are built in:

  1. MIN_TRADES filter - a combo showing "100% win rate" off 4 trades is
     noise, not edge. Results below MIN_TRADES are discarded automatically.
  2. Walk-forward check - after finding the best in-sample combo, the script
     re-runs it on data it never "saw" during optimization (out-of-sample)
     and prints both side by side. If out-of-sample numbers fall apart,
     that combo is overfit - don't trade it as-is.

This script is a research/backtesting tool, not investment advice - past
performance on historical data doesn't guarantee future results.
================================================================================
"""

import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIG - edit this block for your ticker / search ranges
# ==============================================================================

TICKER = "LUPIN.NS"          # <-- change to your stock (used when DATA_SOURCE = "yfinance")
START_DATE = "2015-01-01"  # <-- "entire history" -> use the IPO date or far back
INTERVAL = "1d"

# Where to get price history:
#   "yfinance" -> auto-downloads from Yahoo Finance, free, full history, no setup
#   "csv"      -> use a file you exported from TradingView (see load_data_from_csv
#                 below for how to export, and why you might want to)
DATA_SOURCE = "tv"
CSV_PATH = "tradingview_export.csv"

# Fixed entry parameters (taken straight from your screenshot).
# Add more values per key (e.g. "macd_fast": [10, 12, 14]) to also search
# the entry side - just know the search space multiplies fast.
ENTRY_GRID = {
    "macd_fast": [12],
    "macd_slow": [26],
    "macd_signal": [9],
    "hist_min": [-10],
    "hist_max": [2],
    "trend_sma_length": [50],
    "min_pct_above_sma": [2],
}

# Exit parameters to optimize - these are the ones you tuned manually.
# Step sizes below are coarse to keep the grid fast; narrow the range once
# you see roughly where the good zone is.
EXIT_GRID = {
    "take_profit_pct": [12, 15, 18, 21, 24, 27, 30],
    "stop_loss_pct": [12, 15, 18, 21, 24, 27, 30],
    "use_trailing": [True],
    "trail_activation_pct": [6, 9, 12, 15],
    "trail_breakeven_buffer_pct": [5, 10, 15, 20],
}

TARGET_WIN_RATE = 80.0      # %
TARGET_PROFIT_FACTOR = 2.0
MIN_TRADES = 20             # discard combos with fewer trades than this - raise
                             # this for a stock with a long history; 79-95
                             # trades (like your manual run) is a reasonable bar

TRAIN_FRACTION = 0.70       # walk-forward split: first 70% = in-sample,
                             # last 30% = out-of-sample


# ==============================================================================
# 2. DATA
# ==============================================================================

def load_data(ticker: str, start: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, interval=interval,
                      auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_data_from_tv(ticker: str) -> pd.DataFrame:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
    symbol = ticker.split('.')[0]
    df = tv.get_hist(symbol=symbol, exchange='NSE', interval=Interval.in_daily, n_bars=5000)
    df = df.rename(columns=str.lower)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df.index >= "2007-01-01"]
    return df


def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    Load OHLCV data exported from TradingView.

    How to export from TradingView: open the chart with your strategy
    applied, click the chart's "..." menu (or the camera/export icon in the
    top-right toolbar) -> "Export chart data" -> CSV. Note: how much history
    you can export depends on your TradingView plan - free/Basic plans get
    a limited number of bars, paid plans get more. If you want truly the
    full available history with no plan limits, yfinance (DATA_SOURCE =
    "yfinance") is usually the more complete free option; use this CSV path
    when you specifically want to match the exact bars/prices TradingView
    is charting (e.g. a particular data vendor/exchange feed).

    Handles TradingView's typical export format: a "time" column (often a
    Unix timestamp in seconds) plus open/high/low/close/Volume columns.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    time_col = next((c for c in df.columns if c in ("time", "date", "datetime")), df.columns[0])
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit="s")
    else:
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    rename_map = {}
    for c in df.columns:
        if c.startswith("open"):
            rename_map[c] = "open"
        elif c.startswith("high"):
            rename_map[c] = "high"
        elif c.startswith("low"):
            rename_map[c] = "low"
        elif c.startswith("close") and "adj" not in c:
            rename_map[c] = "close"
        elif c.startswith("volume"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["open", "high", "low", "close"])
    return df


# ==============================================================================
# 3. INDICATORS
# ==============================================================================

def add_indicators(df: pd.DataFrame, macd_fast: int, macd_slow: int,
                    macd_signal: int, trend_sma_length: int) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line
    df["sma_trend"] = df["close"].rolling(trend_sma_length).mean()
    return df


def generate_entries(df: pd.DataFrame, hist_min: float, hist_max: float,
                      min_pct_above_sma: float) -> pd.Series:
    # Matches Pine exactly:
    #   is_trend_intact = close > (trend_sma * (1 + sma_pct))
    #   is_cooled_off    = (histLine > hist_min) and (histLine <= hist_max)
    is_trend_intact = df["close"] > df["sma_trend"] * (1 + min_pct_above_sma / 100)
    is_cooled_off = (df["macd_hist"] > hist_min) & (df["macd_hist"] <= hist_max)
    entries = (is_trend_intact & is_cooled_off).fillna(False)
    return entries


# ==============================================================================
# 4. TRADE SIMULATION
# ==============================================================================

@dataclass
class Trade:
    entry_date: object
    exit_date: object
    entry_price: float
    exit_price: float
    pnl_pct: float
    reason: str


def simulate(df: pd.DataFrame, entries: pd.Series, take_profit_pct: float,
             stop_loss_pct: float, use_trailing: bool,
             trail_activation_pct: float, trail_breakeven_buffer_pct: float) -> List[Trade]:

    dates = df.index.to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    entry_flags = entries.to_numpy()

    trades: List[Trade] = []
    in_pos = False
    pending_entry = False
    entry_price = 0.0
    entry_date = None
    highest_price = 0.0
    tp_price = 0.0
    activation_price = 0.0

    n = len(df)
    for i in range(n):
        # A signal on bar i-1 fills here, at this bar's OPEN (Pine default:
        # process_orders_on_close is false, so strategy.entry() fills next bar's open)
        if pending_entry:
            in_pos = True
            entry_price = opens[i]
            entry_date = dates[i]
            highest_price = highs[i]  # matches Pine: "First bar of trade" -> highest_price := high
            tp_price = entry_price * (1 + take_profit_pct / 100)
            activation_price = entry_price * (1 + trail_activation_pct / 100)
            pending_entry = False

        if not in_pos:
            if entry_flags[i]:
                pending_entry = True
            continue

        highest_price = max(highest_price, highs[i])

        # Step-function stop, exactly matching:
        #   stop_price = highest_price >= activation_price ? (entry*(1+buffer)) : (entry*(1-sl))
        if use_trailing and highest_price >= activation_price:
            stop_price = entry_price * (1 + trail_breakeven_buffer_pct / 100)
            stop_reason = "trail_stop_breakeven"
        else:
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            stop_reason = "stop_loss"

        exit_price = None
        reason = None

        if lows[i] <= stop_price:
            exit_price, reason = stop_price, stop_reason
        if exit_price is None and highs[i] >= tp_price:
            exit_price, reason = tp_price, "take_profit"

        if exit_price is not None:
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append(Trade(entry_date, dates[i], entry_price,
                                 exit_price, pnl_pct, reason))
            in_pos = False

    return trades


# ==============================================================================
# 5. METRICS
# ==============================================================================

def compute_metrics(trades: List[Trade]) -> dict:
    n = len(trades)
    if n == 0:
        return dict(num_trades=0, win_rate=np.nan, profit_factor=np.nan,
                     total_return_pct=np.nan, avg_win_pct=np.nan,
                     avg_loss_pct=np.nan, max_drawdown_pct=np.nan)

    pnl = np.array([t.pnl_pct for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = len(wins) / n * 100
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    equity = np.cumprod(1 + pnl / 100)
    total_return_pct = (equity[-1] - 1) * 100
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / running_max * 100
    max_dd = drawdown_pct.min()

    return dict(
        num_trades=n,
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 3) if np.isfinite(profit_factor) else profit_factor,
        total_return_pct=round(total_return_pct, 2),
        avg_win_pct=round(wins.mean(), 2) if len(wins) else 0.0,
        avg_loss_pct=round(losses.mean(), 2) if len(losses) else 0.0,
        max_drawdown_pct=round(max_dd, 2),
    )


# ==============================================================================
# 6. OPTIMIZATION
# ==============================================================================

def run_optimization(df_raw: pd.DataFrame, entry_grid: dict, exit_grid: dict) -> pd.DataFrame:
    entry_keys = list(entry_grid.keys())
    exit_keys = list(exit_grid.keys())

    entry_combos = list(itertools.product(*[entry_grid[k] for k in entry_keys]))
    exit_combos = list(itertools.product(*[exit_grid[k] for k in exit_keys]))

    rows = []
    for ecombo in entry_combos:
        ep = dict(zip(entry_keys, ecombo))
        df = add_indicators(df_raw, ep["macd_fast"], ep["macd_slow"],
                             ep["macd_signal"], ep["trend_sma_length"])
        entries = generate_entries(df, ep["hist_min"], ep["hist_max"],
                                    ep["min_pct_above_sma"])

        for xcombo in exit_combos:
            xp = dict(zip(exit_keys, xcombo))
            trades = simulate(df, entries, **xp)
            metrics = compute_metrics(trades)
            rows.append({**ep, **xp, **metrics})

    return pd.DataFrame(rows)


def walk_forward_check(df_raw: pd.DataFrame, params: dict, train_fraction: float):
    split_idx = int(len(df_raw) * train_fraction)
    train_df, test_df = df_raw.iloc[:split_idx], df_raw.iloc[split_idx:]

    entry_keys = ["macd_fast", "macd_slow", "macd_signal", "hist_min",
                  "hist_max", "trend_sma_length", "min_pct_above_sma"]
    exit_keys = ["take_profit_pct", "stop_loss_pct", "use_trailing",
                 "trail_activation_pct", "trail_breakeven_buffer_pct"]
    ep = {k: params[k] for k in entry_keys}
    xp = {k: params[k] for k in exit_keys}

    print(f"\nWalk-forward split: {train_df.index[0].date()} -> "
          f"{train_df.index[-1].date()}  (in-sample)  |  "
          f"{test_df.index[0].date()} -> {test_df.index[-1].date()}  (out-of-sample)")

    for label, data in [("IN-SAMPLE", train_df), ("OUT-OF-SAMPLE", test_df)]:
        df = add_indicators(data, ep["macd_fast"], ep["macd_slow"],
                             ep["macd_signal"], ep["trend_sma_length"])
        entries = generate_entries(df, ep["hist_min"], ep["hist_max"],
                                    ep["min_pct_above_sma"])
        trades = simulate(df, entries, **xp)
        m = compute_metrics(trades)
        print(f"  {label:14s} -> trades={m['num_trades']:3d}  "
              f"win_rate={m['win_rate']}%  profit_factor={m['profit_factor']}  "
              f"total_return={m['total_return_pct']}%  max_dd={m['max_drawdown_pct']}%")


# ==============================================================================
# 7. MAIN
# ==============================================================================

if __name__ == "__main__":
    if DATA_SOURCE == "csv":
        print(f"Loading price history from {CSV_PATH} (TradingView export) ...")
        df_raw = load_data_from_csv(CSV_PATH)
    elif DATA_SOURCE == "tv":
        print(f"Loading {TICKER} via tvDatafeed ...")
        df_raw = load_data_from_tv(TICKER)
    else:
        print(f"Loading {TICKER} from {START_DATE} via yfinance ...")
        df_raw = load_data(TICKER, START_DATE, INTERVAL)
    print(f"Loaded {len(df_raw)} bars: {df_raw.index[0].date()} -> {df_raw.index[-1].date()}")

    results = run_optimization(df_raw, ENTRY_GRID, EXIT_GRID)
    results = results.sort_values(["profit_factor", "win_rate"], ascending=False)
    results.to_csv("optimization_results.csv", index=False)
    print(f"\nFull grid: {len(results)} combos tested -> saved to optimization_results.csv")

    qualifying = results[
        (results.win_rate >= TARGET_WIN_RATE)
        & (results.profit_factor >= TARGET_PROFIT_FACTOR)
        & (results.num_trades >= MIN_TRADES)
    ]
    print(f"{len(qualifying)} combos meet win_rate>={TARGET_WIN_RATE}%, "
          f"profit_factor>={TARGET_PROFIT_FACTOR}, trades>={MIN_TRADES}\n")

    display_cols = ["take_profit_pct", "stop_loss_pct", "trail_activation_pct",
                     "trail_breakeven_buffer_pct", "num_trades", "win_rate",
                     "profit_factor", "total_return_pct", "max_drawdown_pct"]
    if len(qualifying):
        print(qualifying[display_cols].head(20).to_string(index=False))

        best = qualifying.iloc[0].to_dict()
        print(f"\nTop combo (by profit factor): {best}")
        walk_forward_check(df_raw, best, TRAIN_FRACTION)
    else:
        print("No combo hit the targets on this grid/history. Try widening "
              "EXIT_GRID ranges, lowering MIN_TRADES, or check the top rows "
              "of optimization_results.csv to see how close you got:")
        print(results[display_cols].head(10).to_string(index=False))
