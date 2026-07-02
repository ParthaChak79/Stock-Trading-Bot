"""
Step 3: pool trades across the whole 45-stock universe under one shared exit rule.

Entry logic is UNCHANGED (MACD cooldown + SMA50 trend). Indicators / ATR / entry
signals don't depend on the exit multiples, so we compute them ONCE per stock
(`prepare_universe`) and cache the arrays. Each candidate exit rule then just
re-runs the fast ATR simulation across every stock and concatenates the trades
into a single pooled list (`pooled_backtest`), tagged by symbol + entry date so
the pooled set can later be split by calendar time and broken back down per stock.
"""

import numpy as np

from wfo.data import get_ohlcv, load_config
from wfo.strategy import (add_atr, add_indicators, entry_signals, r_metrics,
                          _simulate_atr)

DEFAULT_ENTRY = dict(macd_fast=12, macd_slow=26, macd_signal=9,
                     hist_min=-10.0, hist_max=2.0, sma_len=50, sma_pct=0.0)


def prepare_universe(symbols=None, entry_params=None, atr_period=14):
    """Precompute per-stock arrays (OHLC, ATR, entry flags, dates) once."""
    ep = {**DEFAULT_ENTRY, **(entry_params or {})}
    cfg = load_config()
    symbols = symbols or list(cfg.keys())
    universe, skipped = [], []
    for sym in symbols:
        try:
            df = get_ohlcv(sym, cfg[sym])
        except Exception as exc:  # noqa: BLE001
            skipped.append((sym, str(exc)))
            continue
        hist, sma = add_indicators(df, ep["macd_fast"], ep["macd_slow"],
                                   ep["macd_signal"], ep["sma_len"])
        atr = add_atr(df, atr_period)
        entries = entry_signals(df, hist, sma, ep["hist_min"], ep["hist_max"],
                                ep["sma_pct"])
        universe.append(dict(
            symbol=sym,
            opens=df["open"].to_numpy(np.float64),
            highs=df["high"].to_numpy(np.float64),
            lows=df["low"].to_numpy(np.float64),
            atr=atr.to_numpy(np.float64),
            entries=entries,
            dates=df.index.values,          # datetime64[ns]
        ))
    return universe, skipped


def pooled_backtest(universe, tp_atr_mult, sl_atr_mult,
                    trail_act_atr_mult, trail_buf_atr_mult, use_trailing=True):
    """Run one exit rule across all stocks; return the pooled trade record."""
    pnl_r, pnl_pct, entry_dates, syms = [], [], [], []
    per_stock_counts = {}
    for u in universe:
        tr = _simulate_atr(u["opens"], u["highs"], u["lows"], u["atr"],
                           u["entries"], tp_atr_mult, sl_atr_mult,
                           trail_act_atr_mult, trail_buf_atr_mult, use_trailing)
        e_idx, ppct, pr = tr[0], tr[4], tr[5]
        per_stock_counts[u["symbol"]] = int(len(pr))
        if len(pr):
            pnl_r.append(pr)
            pnl_pct.append(ppct)
            entry_dates.append(u["dates"][e_idx])
            syms.append(np.full(len(pr), u["symbol"], dtype=object))

    if pnl_r:
        pnl_r = np.concatenate(pnl_r)
        pnl_pct = np.concatenate(pnl_pct)
        entry_dates = np.concatenate(entry_dates)
        syms = np.concatenate(syms)
    else:
        pnl_r = np.array([])
        pnl_pct = np.array([])
        entry_dates = np.array([], dtype="datetime64[ns]")
        syms = np.array([], dtype=object)

    return dict(pnl_r=pnl_r, pnl_pct=pnl_pct, entry_dates=entry_dates,
                symbols=syms, r_stats=r_metrics(pnl_r),
                per_stock_counts=per_stock_counts)


# Illustrative multiples (mechanics demo only — not optimized)
DEMO = dict(tp_atr_mult=6.0, sl_atr_mult=3.0,
            trail_act_atr_mult=4.0, trail_buf_atr_mult=1.0)


if __name__ == "__main__":
    print("Preparing universe (indicators + ATR + entries per stock) ...")
    universe, skipped = prepare_universe()
    print(f"Universe: {len(universe)} stocks prepared"
          + (f", {len(skipped)} skipped: {skipped}" if skipped else ""))

    pooled = pooled_backtest(universe, **DEMO)
    rs = pooled["r_stats"]
    print(f"\nPooled trades under DEMO rule {DEMO}:")
    print(f"  TOTAL pooled trades : {rs['num_trades']}")
    print(f"  win rate            : {rs['win_rate']}%")
    print(f"  profit factor (R)   : {rs['profit_factor_r']}")
    print(f"  expectancy          : {rs['expectancy_r']} R/trade")
    print(f"  total               : {rs['total_r']} R")
    dts = pooled["entry_dates"]
    if len(dts):
        print(f"  entry-date span     : {np.datetime_as_string(dts.min(), unit='D')} "
              f"-> {np.datetime_as_string(dts.max(), unit='D')}")

    counts = pooled["per_stock_counts"]
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n  per-stock trade counts (top 8): "
          + ", ".join(f"{s}={c}" for s, c in top[:8]))
    print(f"  stocks with >= 15 trades: "
          f"{sum(1 for c in counts.values() if c >= 15)}/{len(counts)}")
