"""
AI Wealth Builder strategy engine (faithful port of the Pine Script v6 logic).

Do NOT "improve" the logic here — this is a straight translation of the
Pine strategy, matching the already-TradingView-validated implementation in
../strategy_optimizer.py:

ENTRY (long only, enter when flat, fills at NEXT bar's open per Pine default):
  - MACD histogram (fast=12, slow=26, signal=9):  hist_min < hist <= hist_max
  - close > SMA(sma_len) * (1 + sma_pct)

EXIT (stop checked before target on the same bar = conservative):
  - take-profit at entry * (1 + tp_pct)
  - if use_trailing and the high since entry has reached entry*(1+trail_activation_pct),
    the stop ratchets ONE TIME to entry*(1 + trail_offset_pct) (breakeven+buffer)
    and stays fixed there; otherwise the stop is entry*(1 - sl_pct).

The path-dependent one-time ratchet does not map to any vectorbt built-in stop,
so the trade path is simulated in a numba loop (fast + exact). The resulting
fills are also wrapped into a vectorbt Portfolio (`to_portfolio`) for
vectorbt-native stats and the parameter-sensitivity heatmaps used later.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit


# ---------------------------------------------------------------------------
# Indicators — computed to match Pine's ta.macd (EMA) and ta.sma exactly.
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, macd_fast: int = 12, macd_slow: int = 26,
                   macd_signal: int = 9, sma_len: int = 50):
    close = df["close"]
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    hist = macd_line - signal_line
    sma = close.rolling(sma_len).mean()
    return hist, sma


def entry_signals(df: pd.DataFrame, hist: pd.Series, sma: pd.Series,
                  hist_min: float = -10.0, hist_max: float = 2.0,
                  sma_pct: float = 0.0) -> np.ndarray:
    # sma_pct is a FRACTION (Pine divides the % input by 100); default 0.0.
    is_cooled = (hist > hist_min) & (hist <= hist_max)
    is_trend = df["close"] > sma * (1.0 + sma_pct)
    return (is_cooled & is_trend).fillna(False).to_numpy()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(period) via Wilder's RMA of True Range (matches TradingView ta.atr).

    True Range = max(high-low, |high-prevclose|, |low-prevclose|). RMA is seeded
    with the SMA of the first `period` TRs, then smoothed recursively:
    RMA_t = (RMA_{t-1}*(period-1) + TR_t) / period.
    """
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))

    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()          # SMA seed
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=df.index)


# ---------------------------------------------------------------------------
# Trade path simulation (numba) — mirrors the validated simulate() exactly.
# ---------------------------------------------------------------------------
@njit
def _simulate(opens, highs, lows, entry_flags, tp_pct, sl_pct,
              use_trailing, trail_activation_pct, trail_offset_pct):
    n = opens.shape[0]
    entry_idx = np.empty(n, dtype=np.int64)
    exit_idx = np.empty(n, dtype=np.int64)
    entry_px = np.empty(n, dtype=np.float64)
    exit_px = np.empty(n, dtype=np.float64)
    pnl = np.empty(n, dtype=np.float64)
    k = 0

    in_pos = False
    pending = False
    ep = 0.0
    highest = 0.0
    tp_price = 0.0
    activation = 0.0
    cur_entry_idx = 0

    for i in range(n):
        # A signal on bar i-1 fills here at bar i's OPEN (Pine default fill).
        if pending:
            in_pos = True
            ep = opens[i]
            cur_entry_idx = i
            highest = highs[i]              # Pine: first bar of trade -> highest := high
            tp_price = ep * (1.0 + tp_pct)
            activation = ep * (1.0 + trail_activation_pct)
            pending = False

        if not in_pos:
            if entry_flags[i]:
                pending = True
            continue

        if highs[i] > highest:
            highest = highs[i]

        # One-time breakeven ratchet: once high hit activation, stop locks at buffer.
        if use_trailing and highest >= activation:
            stop_price = ep * (1.0 + trail_offset_pct)
        else:
            stop_price = ep * (1.0 - sl_pct)

        ex = np.nan
        if lows[i] <= stop_price:          # stop checked first (conservative)
            ex = stop_price
        if np.isnan(ex) and highs[i] >= tp_price:
            ex = tp_price

        if not np.isnan(ex):
            entry_idx[k] = cur_entry_idx
            exit_idx[k] = i
            entry_px[k] = ep
            exit_px[k] = ex
            pnl[k] = (ex - ep) / ep * 100.0
            k += 1
            in_pos = False

    return entry_idx[:k], exit_idx[:k], entry_px[:k], exit_px[:k], pnl[:k]


# ---------------------------------------------------------------------------
# Metrics (matches ../strategy_optimizer.py compute_metrics).
# ---------------------------------------------------------------------------
def compute_metrics(pnl: np.ndarray) -> dict:
    n = len(pnl)
    if n == 0:
        return dict(num_trades=0, win_rate=np.nan, profit_factor=np.nan,
                    profit_factor_pct=np.nan, total_return_pct=np.nan,
                    max_drawdown_pct=np.nan)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n * 100.0

    # profit_factor -> TradingView's definition: gross profit / gross loss in
    # CURRENCY, with position size compounding at 100% of equity. PF is fully
    # determined by the ordered trade-% sequence under this compounding.
    equity = 1.0
    gross_profit = 0.0
    gross_loss = 0.0
    eq_curve = np.empty(n)
    for i in range(n):
        d = equity * (pnl[i] / 100.0)
        if d > 0:
            gross_profit += d
        else:
            gross_loss += -d
        equity *= (1 + pnl[i] / 100.0)
        eq_curve[i] = equity
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Secondary: equal-weight percentage-sum PF (position-size-independent).
    gp_pct = wins.sum() if len(wins) else 0.0
    gl_pct = abs(losses.sum()) if len(losses) else 0.0
    profit_factor_pct = (gp_pct / gl_pct) if gl_pct > 0 else np.inf

    total_return_pct = (eq_curve[-1] - 1) * 100.0
    running_max = np.maximum.accumulate(eq_curve)
    max_dd = ((eq_curve - running_max) / running_max * 100.0).min()
    return dict(
        num_trades=n,
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 3) if np.isfinite(profit_factor) else np.inf,
        profit_factor_pct=round(profit_factor_pct, 3) if np.isfinite(profit_factor_pct) else np.inf,
        total_return_pct=round(total_return_pct, 2),
        max_drawdown_pct=round(max_dd, 2),
    )


def backtest(df: pd.DataFrame, tp_pct: float, sl_pct: float,
             trail_activation_pct: float, trail_offset_pct: float,
             use_trailing: bool = True,
             macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
             hist_min: float = -10.0, hist_max: float = 2.0,
             sma_len: int = 50, sma_pct: float = 0.0):
    """Run one backtest. Returns (metrics_dict, trades_tuple)."""
    hist, sma = add_indicators(df, macd_fast, macd_slow, macd_signal, sma_len)
    entries = entry_signals(df, hist, sma, hist_min, hist_max, sma_pct)
    trades = _simulate(
        df["open"].to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), entries,
        tp_pct, sl_pct, use_trailing, trail_activation_pct, trail_offset_pct)
    return compute_metrics(trades[4]), trades


@njit
def _simulate_atr(opens, highs, lows, atr, entry_flags,
                  tp_m, sl_m, trail_act_m, trail_buf_m, use_trailing):
    """Same path logic as _simulate, but exit levels are ATR multiples off entry.

    ATR at entry = ATR as of the SIGNAL bar (known before the next-bar-open fill).
    Returns per-trade arrays incl. pnl in % AND in R (R = sl_m * ATR_at_entry).
    """
    n = opens.shape[0]
    e_idx = np.empty(n, dtype=np.int64)
    x_idx = np.empty(n, dtype=np.int64)
    e_px = np.empty(n, dtype=np.float64)
    x_px = np.empty(n, dtype=np.float64)
    pnl_pct = np.empty(n, dtype=np.float64)
    pnl_r = np.empty(n, dtype=np.float64)
    atr_e_arr = np.empty(n, dtype=np.float64)
    k = 0

    in_pos = False
    pending = False
    pend_atr = 0.0
    ep = highest = tp_price = sl_price = act_price = trail_stop = R = 0.0
    cur_atr_e = 0.0
    cur_entry = 0

    for i in range(n):
        if pending:
            ep = opens[i]
            cur_entry = i
            highest = highs[i]
            atr_e = pend_atr
            tp_price = ep + tp_m * atr_e
            sl_price = ep - sl_m * atr_e
            act_price = ep + trail_act_m * atr_e
            trail_stop = ep + trail_buf_m * atr_e
            R = sl_m * atr_e
            cur_atr_e = atr_e
            in_pos = True
            pending = False

        if not in_pos:
            # arm entry only when ATR at the signal bar is valid
            if entry_flags[i] and atr[i] > 0.0 and atr[i] == atr[i]:
                pending = True
                pend_atr = atr[i]
            continue

        if highs[i] > highest:
            highest = highs[i]

        if use_trailing and highest >= act_price:
            stop = trail_stop
        else:
            stop = sl_price

        ex = np.nan
        if lows[i] <= stop:                # stop checked first (conservative)
            ex = stop
        if np.isnan(ex) and highs[i] >= tp_price:
            ex = tp_price

        if not np.isnan(ex):
            e_idx[k] = cur_entry
            x_idx[k] = i
            e_px[k] = ep
            x_px[k] = ex
            pnl_pct[k] = (ex - ep) / ep * 100.0
            pnl_r[k] = (ex - ep) / R if R > 0 else 0.0
            atr_e_arr[k] = cur_atr_e
            k += 1
            in_pos = False

    return (e_idx[:k], x_idx[:k], e_px[:k], x_px[:k],
            pnl_pct[:k], pnl_r[:k], atr_e_arr[:k])


def r_metrics(pnl_r: np.ndarray) -> dict:
    """R-multiple pooled metrics: PF, expectancy, win rate (equal-weight/trade)."""
    n = len(pnl_r)
    if n == 0:
        return dict(num_trades=0, win_rate=np.nan, profit_factor_r=np.nan,
                    expectancy_r=np.nan, total_r=np.nan)
    wins = pnl_r[pnl_r > 0]
    losses = pnl_r[pnl_r <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    return dict(
        num_trades=n,
        win_rate=round(len(wins) / n * 100.0, 2),
        profit_factor_r=round(pf, 3) if np.isfinite(pf) else np.inf,
        expectancy_r=round(float(pnl_r.mean()), 4),
        total_r=round(float(pnl_r.sum()), 2),
    )


def backtest_atr(df: pd.DataFrame, tp_atr_mult: float, sl_atr_mult: float,
                 trail_act_atr_mult: float, trail_buf_atr_mult: float,
                 use_trailing: bool = True, atr_period: int = 14,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 hist_min: float = -10.0, hist_max: float = 2.0,
                 sma_len: int = 50, sma_pct: float = 0.0):
    """ATR-multiple exits. Returns (metrics, r_stats, trades).

    metrics = %-based (compute_metrics, for continuity with the old numbers);
    r_stats = R-multiple pooled metrics; trades tuple carries pnl_r + atr_at_entry.
    """
    hist, sma = add_indicators(df, macd_fast, macd_slow, macd_signal, sma_len)
    atr = add_atr(df, atr_period)
    entries = entry_signals(df, hist, sma, hist_min, hist_max, sma_pct)
    trades = _simulate_atr(
        df["open"].to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), atr.to_numpy(np.float64), entries,
        tp_atr_mult, sl_atr_mult, trail_act_atr_mult, trail_buf_atr_mult,
        use_trailing)
    return compute_metrics(trades[4]), r_metrics(trades[5]), trades


def to_portfolio(df: pd.DataFrame, trades, init_cash: float = 100000.0):
    """Wrap simulated fills into a vectorbt Portfolio for native stats/heatmaps.

    Primary metrics come from compute_metrics (validated); this is the vectorbt
    view for cross-checking and later analysis. Same-bar entry+exit trades (rare)
    are placed one bar apart so from_orders can represent both legs.
    """
    e_idx, x_idx, e_px, x_px, _ = trades
    close = df["close"]
    n = len(close)
    size = np.full(n, np.nan)
    price = np.full(n, np.nan)
    for ei, xi, ep, xp in zip(e_idx, x_idx, e_px, x_px):
        if xi <= ei:
            xi = min(ei + 1, n - 1)
        size[ei] = np.inf       # buy with all cash (percent_of_equity=100)
        price[ei] = ep
        size[xi] = -np.inf      # close the position
        price[xi] = xp
    return vbt.Portfolio.from_orders(
        close, size=size, price=price, init_cash=init_cash, freq="1D",
        direction="longonly")   # -inf must only CLOSE the long, never open a short
