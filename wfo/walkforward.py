"""
Walk-forward optimization harness (single price series).

Splits history into N rolling windows. Each window is a contiguous slice of
`window_frac` of the data; within it the first `train_frac` is the TRAIN period
and the remainder is the TEST (out-of-sample) period. Window start rolls forward
evenly from the start of history to the end, so each test block probes a
different, later out-of-sample stretch.

For every window:
  - run an Optuna study on the TRAIN slice (maximize guarded profit factor)
  - take the best train params and evaluate them, untouched, on the TEST slice
  - record profit factor / win rate / trades / max drawdown for BOTH train (IS)
    and test (OOS)

The gap between IS and OOS profit factor across windows is the core overfitting
signal, consumed by diagnostics.py.
"""

import numpy as np

from wfo.optimize import optimize, best_params, METRIC_KEYS
from wfo.strategy import backtest


def make_windows(n_bars, n_windows=4, train_frac=0.70, window_frac=0.5):
    """Return list of (train_start, split, window_end) index triples."""
    W = int(n_bars * window_frac)
    if n_windows == 1:
        starts = [0]
    else:
        starts = [round(i * (n_bars - W) / (n_windows - 1)) for i in range(n_windows)]
    windows = []
    for s in starts:
        e = s + W
        split = s + int(W * train_frac)
        windows.append((s, split, e))
    return windows


def walk_forward(df, n_windows=4, train_frac=0.70, window_frac=0.5,
                 n_trials=150, min_trades=10, seed=42, entry_params=None):
    n = len(df)
    windows = make_windows(n, n_windows, train_frac, window_frac)
    results = []
    for wi, (s, split, e) in enumerate(windows):
        train_df = df.iloc[s:split]
        test_df = df.iloc[split:e]

        study = optimize(train_df, n_trials=n_trials, seed=seed,
                         min_trades=min_trades, entry_params=entry_params)
        bp = best_params(study)
        valid = bp.get("valid", True)
        params = dict(tp_pct=bp["tp_pct"], sl_pct=bp["sl_pct"],
                      trail_activation_pct=bp["trail_activation_pct"],
                      trail_offset_pct=bp["trail_offset_pct"])

        # If no train trial passed the guard, this window yields no usable params.
        is_metrics = {k: bp.get(k) for k in METRIC_KEYS} if valid else None
        oos_metrics = None
        if valid:
            oos_metrics, _ = backtest(test_df, **params, **(entry_params or {}))

        results.append(dict(
            window=wi,
            valid=valid,
            train_start=str(df.index[s].date()),
            train_end=str(df.index[split - 1].date()),
            test_start=str(df.index[split].date()),
            test_end=str(df.index[e - 1].date()),
            params=params if valid else None,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
        ))
    return results


def make_windows_pooled(n_bars, n_windows=4, train_frac=0.70):
    """Non-overlapping test blocks so OOS trades can be pooled without double-count.

    The first `train_frac` of history seeds the initial train span; the remaining
    tail is split into `n_windows` contiguous, non-overlapping TEST blocks. Each
    test block is preceded by a rolling train window of length `train_len`
    (= the initial train span). Returns (train_start, test_start, test_end) triples.
    """
    test_region_start = int(n_bars * train_frac)
    total_test = n_bars - test_region_start
    test_size = max(1, total_test // n_windows)
    train_len = test_region_start
    windows = []
    for i in range(n_windows):
        test_start = test_region_start + i * test_size
        test_end = n_bars if i == n_windows - 1 else test_start + test_size
        if test_start >= n_bars:
            break
        train_start = max(0, test_start - train_len)
        windows.append((train_start, test_start, test_end))
    return windows


def walk_forward_pooled(df, n_windows=4, train_frac=0.70, n_trials=150,
                        min_trades=10, seed=42, entry_params=None):
    """Walk forward over NON-overlapping test blocks and pool the OOS trades.

    Returns dict(per_window=[...], pooled_oos_pnl=ndarray). Each per_window entry
    is shaped for analyze_gap (valid / is_metrics / oos_metrics), plus the raw
    OOS trade PnLs for that block.
    """
    n = len(df)
    windows = make_windows_pooled(n, n_windows, train_frac)
    per_window = []
    pooled = []
    for wi, (ts, split, te) in enumerate(windows):
        train_df = df.iloc[ts:split]
        test_df = df.iloc[split:te]
        bp = best_params(optimize(train_df, n_trials=n_trials, seed=seed,
                                  min_trades=min_trades, entry_params=entry_params))
        valid = bp.get("valid", True)
        params = dict(tp_pct=bp["tp_pct"], sl_pct=bp["sl_pct"],
                      trail_activation_pct=bp["trail_activation_pct"],
                      trail_offset_pct=bp["trail_offset_pct"])
        is_metrics = {k: bp.get(k) for k in METRIC_KEYS} if valid else None
        oos_metrics = None
        oos_pnl = np.array([])
        if valid:
            oos_metrics, trades = backtest(test_df, **params, **(entry_params or {}))
            oos_pnl = trades[4]
            if len(oos_pnl):
                pooled.append(oos_pnl)
        per_window.append(dict(
            window=wi, valid=valid,
            train_start=str(df.index[ts].date()), train_end=str(df.index[split - 1].date()),
            test_start=str(df.index[split].date()), test_end=str(df.index[te - 1].date()),
            params=params if valid else None,
            is_metrics=is_metrics, oos_metrics=oos_metrics, oos_n=int(len(oos_pnl))))
    pooled_arr = np.concatenate(pooled) if pooled else np.array([])
    return dict(per_window=per_window, pooled_oos_pnl=pooled_arr)


def _fmt(m):
    return (f"trades={int(m['num_trades']) if m['num_trades'] == m['num_trades'] else 0:>3}  "
            f"win={m['win_rate']}%  PF={m['profit_factor']}  "
            f"ret={m['total_return_pct']}%  dd={m['max_drawdown_pct']}%")


if __name__ == "__main__":
    import sys
    from wfo.data import get_ohlcv, load_config

    symbol = sys.argv[1] if len(sys.argv) > 1 else "NH"
    cfg = load_config()
    entry = cfg[symbol]
    df = get_ohlcv(symbol, entry)
    print(f"=== Walk-forward: {symbol} ({entry.get('name','')}) ===")
    print(f"Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}\n")

    results = walk_forward(df, n_windows=4, train_frac=0.70,
                           window_frac=0.5, n_trials=150)
    for r in results:
        print(f"Window {r['window']}: "
              f"train {r['train_start']}..{r['train_end']}  |  "
              f"test {r['test_start']}..{r['test_end']}")
        if not r["valid"]:
            print("  NO VALID PARAMS: no trial met the min-trades + positive-return "
                  "guard on this train slice (strategy trades too rarely here)\n")
            continue
        p = r["params"]
        print(f"  best params: tp={p['tp_pct']:.3f} sl={p['sl_pct']:.3f} "
              f"trail_act={p['trail_activation_pct']:.3f} trail_buf={p['trail_offset_pct']:.3f}")
        print(f"  IN-SAMPLE : {_fmt(r['is_metrics'])}")
        print(f"  OUT-SAMPLE: {_fmt(r['oos_metrics'])}\n")
