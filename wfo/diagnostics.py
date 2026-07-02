"""
Overfitting diagnostics for the walk-forward optimizer (single stock).

Three diagnostics, per the spec:
  1. IN-SAMPLE vs OUT-OF-SAMPLE profit-factor gap across windows.
     Large degradation (OOS PF collapses vs IS PF) => overfit.
  2. Parameter-sensitivity heatmap: tp_pct x sl_pct with the trailing params held
     at their optimum. A broad plateau of good PF => trustworthy; a lone narrow
     spike => noise/overfit.
  3. Low-trade-count flag: windows whose OOS trade count is below `min_trades`
     can't produce a statistically meaningful profit factor.

Produces a structured verdict (STABLE / UNSTABLE + reasons) and saves the
heatmap as CSV + PNG under wfo/results/.
"""

import os

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt
import numpy as np

from wfo.optimize import optimize, best_params
from wfo.strategy import backtest, compute_metrics
from wfo.walkforward import walk_forward_pooled

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. IS vs OOS profit-factor gap
# ---------------------------------------------------------------------------
def analyze_gap(wf_results, min_trades=10):
    per_window = []
    ratios = []
    n_invalid = n_low_oos = n_degenerate = 0

    for r in wf_results:
        if not r["valid"]:
            n_invalid += 1
            per_window.append(dict(window=r["window"], status="no_valid_params"))
            continue
        is_pf = r["is_metrics"]["profit_factor"]
        oos = r["oos_metrics"]
        oos_pf = oos["profit_factor"]
        oos_trades = oos["num_trades"]
        low = oos_trades < min_trades
        if low:
            n_low_oos += 1
        # inf IS PF (no in-sample losses) is itself an overfit red flag
        degenerate = not np.isfinite(is_pf)
        if degenerate:
            n_degenerate += 1
        ratio = None
        if np.isfinite(is_pf) and np.isfinite(oos_pf) and is_pf > 0:
            ratio = oos_pf / is_pf
            if not low:
                ratios.append(ratio)
        per_window.append(dict(
            window=r["window"], status="ok", is_pf=is_pf, oos_pf=oos_pf,
            oos_trades=int(oos_trades), oos_is_ratio=ratio,
            low_trades=low, degenerate_is=degenerate))

    median_ratio = float(np.median(ratios)) if ratios else None
    # aggregate OOS PF: median across windows that are valid, finite and not low-trade
    oos_pfs = [w["oos_pf"] for w in per_window
               if w["status"] == "ok" and not w["low_trades"]
               and np.isfinite(w["oos_pf"])]
    oos_pf_median = float(np.median(oos_pfs)) if oos_pfs else None
    return dict(
        per_window=per_window,
        n_windows=len(wf_results),
        n_invalid=n_invalid,
        n_low_oos=n_low_oos,
        n_degenerate=n_degenerate,
        median_oos_is_ratio=median_ratio,
        oos_pf_median=oos_pf_median,
        n_ratio_samples=len(ratios),
    )


# ---------------------------------------------------------------------------
# 2. Parameter-sensitivity heatmap (tp x sl, trail params fixed at optimum)
# ---------------------------------------------------------------------------
def sensitivity_heatmap(df, trail_act, trail_buf, symbol="stock",
                        n=9, min_trades=10, entry_params=None,
                        tp_lo=0.10, tp_hi=0.35, sl_lo=0.08, sl_hi=0.32,
                        max_gap=0.15, pf_cap=100.0, save=True):
    entry_params = entry_params or {}
    tp_range = np.linspace(tp_lo, tp_hi, n)
    sl_range = np.linspace(sl_lo, sl_hi, n)
    grid = np.full((len(sl_range), len(tp_range)), np.nan)

    for i, sl in enumerate(sl_range):
        for j, tp in enumerate(tp_range):
            if tp <= trail_act or abs(tp - sl) > max_gap:   # enforce ordering rules
                continue
            m, _ = backtest(df, tp_pct=tp, sl_pct=sl, trail_activation_pct=trail_act,
                            trail_offset_pct=trail_buf, **entry_params)
            if m["num_trades"] < min_trades or not (m["total_return_pct"] > 0):
                continue
            pf = m["profit_factor"]
            grid[i, j] = min(pf, pf_cap) if np.isfinite(pf) else pf_cap

    finite = grid[np.isfinite(grid)]
    n_cells = int(np.isfinite(grid).sum())
    peak = float(finite.max()) if len(finite) else None
    # plateau fraction: of the valid cells, how many are within 20% of the peak
    plateau_frac = float((finite >= 0.8 * peak).sum() / n_cells) if n_cells else 0.0

    csv_path = png_path = None
    if save:
        csv_path = os.path.join(RESULTS_DIR, f"{symbol}_sensitivity.csv")
        _save_grid_csv(csv_path, tp_range, sl_range, grid)
        png_path = os.path.join(RESULTS_DIR, f"{symbol}_sensitivity.png")
        _save_grid_png(png_path, tp_range, sl_range, grid, symbol,
                       trail_act, trail_buf)

    return dict(tp_range=tp_range, sl_range=sl_range, grid=grid,
                peak_pf=peak, n_valid_cells=n_cells, plateau_frac=plateau_frac,
                csv_path=csv_path, png_path=png_path)


def _save_grid_csv(path, tp_range, sl_range, grid):
    with open(path, "w") as f:
        f.write("sl\\tp," + ",".join(f"{tp:.3f}" for tp in tp_range) + "\n")
        for i, sl in enumerate(sl_range):
            row = ",".join("" if not np.isfinite(v) else f"{v:.3f}" for v in grid[i])
            f.write(f"{sl:.3f},{row}\n")


def _save_grid_png(path, tp_range, sl_range, grid, symbol, trail_act, trail_buf):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[tp_range[0], tp_range[-1], sl_range[0], sl_range[-1]])
    fig.colorbar(im, ax=ax, label="Profit factor (capped 100)")
    ax.set_xlabel("take_profit_pct")
    ax.set_ylabel("stop_loss_pct")
    ax.set_title(f"{symbol}: PF sensitivity (tp x sl)\n"
                 f"trail_act={trail_act:.3f}, trail_buf={trail_buf:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Verdict
# ---------------------------------------------------------------------------
def verdict(gap, heat, min_trades=10, plateau_min=0.20,
            ratio_hard=0.40, ratio_soft=0.60):
    """Graded verdict: STABLE / BORDERLINE / UNSTABLE.

    HARD failures (can't trust the result at all) -> UNSTABLE.
    SOFT concerns (usable but caveat) -> BORDERLINE.
    Neither -> STABLE.
    """
    hard, soft = [], []
    n = gap["n_windows"]
    n_unusable = gap["n_invalid"] + gap["n_low_oos"]
    usable = n - n_unusable
    ratio = gap["median_oos_is_ratio"]

    # ---- HARD failures -> UNSTABLE ----
    if usable <= 0:
        hard.append(f"no window has a significant OOS sample (>= {min_trades} trades)")
    if n_unusable > n / 2:
        hard.append(f"{n_unusable}/{n} windows invalid or below {min_trades} OOS trades")
    if ratio is not None and ratio < ratio_hard:
        hard.append(f"OOS profit factor collapses to {ratio:.0%} of in-sample")
    if heat["peak_pf"] is None:
        hard.append("sensitivity map empty (no tp/sl combo meets the guard)")

    # ---- SOFT concerns -> BORDERLINE ----
    if gap["n_degenerate"] > 0:
        soft.append(f"{gap['n_degenerate']} window(s) with infinite in-sample PF "
                    "(no in-sample losses)")
    if ratio is not None and ratio_hard <= ratio < ratio_soft:
        soft.append(f"OOS profit factor degrades to {ratio:.0%} of in-sample")
    if 0 < n_unusable <= n / 2:
        soft.append(f"{n_unusable}/{n} window(s) had too few OOS trades")
    if heat["peak_pf"] is not None and heat["plateau_frac"] < plateau_min:
        soft.append(f"narrow optimum: only {heat['plateau_frac']:.0%} of the tp/sl "
                    "grid is near-peak (spike, not plateau)")

    if hard:
        grade = "UNSTABLE"
    elif soft:
        grade = "BORDERLINE"
    else:
        grade = "STABLE"
    return dict(grade=grade, stable=(grade == "STABLE"),
                reasons=hard + soft, hard=hard, soft=soft)


# ---------------------------------------------------------------------------
# Plateau-based recommendation (robust to narrow spikes)
# ---------------------------------------------------------------------------
def plateau_recommendation(df, heat, trail_act, trail_buf, entry_params=None,
                           near=0.80, min_neighbors=4):
    """Recommend tp/sl at the center of the broadest high-PF plateau.

    Rather than the raw PF peak (which may be an isolated spike), score each
    tp/sl cell by the MEAN PF over its 3x3 neighborhood (needing >= min_neighbors
    valid neighbors) and pick the argmax. Also returns `local_support_frac`: the
    fraction of that cell's neighbors within `near` of its own PF -> how broad the
    plateau is right at the recommendation. Trailing params are held at the
    full-history optimum (same surface as the heatmap).
    """
    entry_params = entry_params or {}
    grid = heat["grid"]
    tp_range = heat["tp_range"]
    sl_range = heat["sl_range"]
    nr, nc = grid.shape

    smoothed = np.full_like(grid, np.nan)
    for i in range(nr):
        for j in range(nc):
            if not np.isfinite(grid[i, j]):
                continue
            vals = [grid[ii, jj]
                    for ii in (i - 1, i, i + 1) for jj in (j - 1, j, j + 1)
                    if 0 <= ii < nr and 0 <= jj < nc and np.isfinite(grid[ii, jj])]
            if len(vals) >= min_neighbors:
                smoothed[i, j] = float(np.mean(vals))

    fallback = False
    if np.isfinite(smoothed).any():
        i, j = np.unravel_index(np.nanargmax(smoothed), smoothed.shape)
    elif np.isfinite(grid).any():
        fallback = True                      # no plateau; fall back to raw peak
        i, j = np.unravel_index(np.nanargmax(grid), grid.shape)
    else:
        return None                          # empty surface

    center = grid[i, j]
    neigh = [grid[ii, jj]
             for ii in (i - 1, i, i + 1) for jj in (j - 1, j, j + 1)
             if 0 <= ii < nr and 0 <= jj < nc and np.isfinite(grid[ii, jj])
             and not (ii == i and jj == j)]
    local_support = (sum(v >= near * center for v in neigh) / len(neigh)
                     if neigh else 0.0)

    rec_tp = round(float(tp_range[j]), 3)
    rec_sl = round(float(sl_range[i]), 3)
    m, _ = backtest(df, tp_pct=rec_tp, sl_pct=rec_sl,
                    trail_activation_pct=trail_act, trail_offset_pct=trail_buf,
                    **entry_params)
    return dict(rec_tp=rec_tp, rec_sl=rec_sl,
                rec_trail_act=round(float(trail_act), 3),
                rec_trail_buf=round(float(trail_buf), 3),
                plateau_pf=None if fallback else round(float(smoothed[i, j]), 3),
                cell_pf=None if not np.isfinite(center) else round(float(center), 3),
                local_support_frac=round(float(local_support), 3),
                is_fallback_peak=fallback, full_metrics=m)


# ---------------------------------------------------------------------------
# Pooled out-of-sample analysis (for low-frequency strategies)
# ---------------------------------------------------------------------------
def analyze_pooled(pooled_wf):
    """Metrics on the concatenated OOS trades across non-overlapping windows.

    This is the actual walk-forward deployment record: re-optimize each period,
    trade the next (unseen) block, pool every OOS trade into one sample. Much
    more statistical power than any single window for a low-frequency strategy.
    """
    pnl = pooled_wf["pooled_oos_pnl"]
    m = compute_metrics(pnl)
    return dict(
        pooled_trades=int(m["num_trades"]),
        pooled_oos_pf=m["profit_factor"],
        pooled_win_rate=m["win_rate"],
        pooled_return_pct=m["total_return_pct"],
        pooled_max_dd=m["max_drawdown_pct"],
    )


def verdict_pooled(pooled, plateau_rec, min_pooled=15, pf_stable=1.5,
                   support_min=0.50):
    """Graded verdict from the pooled OOS record.

    UNSTABLE  : too few pooled OOS trades, or pooled OOS PF < 1 (loses money OOS).
    BORDERLINE: profitable OOS (1 <= PF < pf_stable), or the RECOMMENDED params
                sit on a narrow ridge (few near-peak neighbors).
    STABLE    : pooled OOS PF >= pf_stable on a significant sample AND the
                recommended params sit on a broad plateau.
    """
    n = pooled["pooled_trades"]
    pf = pooled["pooled_oos_pf"]
    hard, soft = [], []

    if n < min_pooled:
        hard.append(f"only {n} pooled OOS trades (< {min_pooled}); not significant")
    if pf is not None and pf != float("inf") and pf < 1.0:
        hard.append(f"pooled OOS profit factor {pf:.2f} < 1.0 (net-losing out-of-sample)")

    if not hard:
        if not (pf is not None and (pf == float("inf") or pf >= pf_stable)):
            soft.append(f"pooled OOS profit factor {pf:.2f} is modest (< {pf_stable})")
        # narrow optimum now judged at the RECOMMENDED (plateau) params, not globally
        if plateau_rec is None:
            soft.append("no valid parameter surface for a plateau recommendation")
        elif plateau_rec["is_fallback_peak"] or plateau_rec["local_support_frac"] < support_min:
            soft.append(f"recommended params sit on a narrow ridge "
                        f"({plateau_rec['local_support_frac']:.0%} of neighbors near-peak)")

    if hard:
        grade = "UNSTABLE"
    elif soft:
        grade = "BORDERLINE"
    else:
        grade = "STABLE"
    return dict(grade=grade, stable=(grade == "STABLE"),
                reasons=hard + soft, hard=hard, soft=soft)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def diagnose(symbol, df, n_windows=4, train_frac=0.70,
             n_trials=150, min_trades=10, entry_params=None, save=True):
    # Non-overlapping walk-forward: powers BOTH the per-window gap analysis and
    # the pooled OOS analysis from one set of studies.
    pooled_wf = walk_forward_pooled(df, n_windows=n_windows, train_frac=train_frac,
                                    n_trials=n_trials, min_trades=min_trades,
                                    entry_params=entry_params)
    gap = analyze_gap(pooled_wf["per_window"], min_trades=min_trades)
    pooled = analyze_pooled(pooled_wf)

    # trail optimum for the heatmap: from a full-history study
    full = best_params(optimize(df, n_trials=n_trials, min_trades=min_trades,
                                entry_params=entry_params))
    heat = sensitivity_heatmap(df, full["trail_activation_pct"],
                               full["trail_offset_pct"], symbol=symbol,
                               min_trades=min_trades, entry_params=entry_params,
                               save=save)
    # robust recommendation = center of the broadest plateau on that surface
    plateau_rec = plateau_recommendation(df, heat, full["trail_activation_pct"],
                                         full["trail_offset_pct"],
                                         entry_params=entry_params)

    v_window = verdict(gap, heat, min_trades=min_trades)
    v_pooled = verdict_pooled(pooled, plateau_rec)
    return dict(symbol=symbol, walk_forward=pooled_wf["per_window"],
                gap=gap, pooled=pooled, heatmap=heat, full_history_best=full,
                plateau_rec=plateau_rec, verdict=v_pooled, verdict_window=v_window)


def print_report(diag):
    s = diag["symbol"]
    g = diag["gap"]
    h = diag["heatmap"]
    v = diag["verdict"]
    print(f"\n===== DIAGNOSTICS: {s} =====")
    print(f"Windows: {g['n_windows']}  |  invalid: {g['n_invalid']}  |  "
          f"low-OOS-trade: {g['n_low_oos']}  |  inf-IS-PF: {g['n_degenerate']}")
    print("IS -> OOS profit factor by window:")
    for w in g["per_window"]:
        if w["status"] != "ok":
            print(f"  W{w['window']}: {w['status']}")
            continue
        ratio = "n/a" if w["oos_is_ratio"] is None else f"{w['oos_is_ratio']:.2f}x"
        note = " [LOW OOS TRADES]" if w["low_trades"] else ""
        note += " [inf IS PF]" if w["degenerate_is"] else ""
        print(f"  W{w['window']}: IS PF={w['is_pf']}  OOS PF={w['oos_pf']}  "
              f"(OOS/IS={ratio}, OOS trades={w['oos_trades']}){note}")
    mr = g["median_oos_is_ratio"]
    print(f"Median OOS/IS PF ratio (significant windows): "
          f"{'n/a' if mr is None else f'{mr:.2f}x'}")
    print(f"Sensitivity: peak PF={h['peak_pf']}  valid cells={h['n_valid_cells']}  "
          f"plateau_frac={h['plateau_frac']:.0%}  -> {h['png_path']}")

    p = diag["pooled"]
    print(f"\nPOOLED OOS (all non-overlapping test blocks combined):")
    print(f"  trades={p['pooled_trades']}  PF={p['pooled_oos_pf']}  "
          f"win={p['pooled_win_rate']}%  return={p['pooled_return_pct']}%  "
          f"maxDD={p['pooled_max_dd']}%")

    pr = diag["plateau_rec"]
    if pr:
        peak = diag["full_history_best"]
        print(f"\nRECOMMENDED PARAMS (plateau center, robust to spikes):")
        print(f"  tp={pr['rec_tp']} sl={pr['rec_sl']} "
              f"trail_act={pr['rec_trail_act']} trail_buf={pr['rec_trail_buf']}  "
              f"(local support={pr['local_support_frac']:.0%}"
              f"{', FELL BACK TO PEAK' if pr['is_fallback_peak'] else ''})")
        print(f"  full-history PF at recommendation: {pr['full_metrics']['profit_factor']} "
              f"(raw peak was tp={peak['tp_pct']:.3f}/sl={peak['sl_pct']:.3f}, "
              f"PF={peak['profit_factor']})")

    print(f"\nVERDICT (pooled OOS): {v['grade']}")
    for r in v["reasons"]:
        print(f"  - {r}")
    vw = diag["verdict_window"]
    print(f"per-window verdict (secondary): {vw['grade']}")


if __name__ == "__main__":
    import sys
    from wfo.data import get_ohlcv, load_config

    symbol = sys.argv[1] if len(sys.argv) > 1 else "NH"
    cfg = load_config()
    entry = cfg[symbol]
    df = get_ohlcv(symbol, entry)
    print(f"Diagnosing {symbol} ({entry.get('name','')}) — "
          f"{len(df)} bars {df.index[0].date()}..{df.index[-1].date()}")
    diag = diagnose(symbol, df)
    print_report(diag)
