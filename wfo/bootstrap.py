"""
Step 6: bootstrap confidence intervals + expectancy-in-R on pooled trades.

Resample the pooled trade R-multiples with replacement (default 2000 iterations)
and report the 5th / 50th / 95th percentile (a 90% CI) of profit factor, win rate,
and expectancy - a confidence RANGE, not a single point estimate. Expectancy in R
catches "high win rate but occasional huge loss" failure modes that PF alone hides.

This module bootstraps whatever pooled trade set it's given: here (Step 6) the
full-history in-sample set, later (after Step 7) the pooled OUT-OF-SAMPLE set.
"""

import numpy as np


def bootstrap_ci(pnl_r, n_iter=2000, seed=42, pf_cap=100.0):
    """Return point + 90% CI (p5/p50/p95) for PF(R), win rate, expectancy."""
    pnl_r = np.asarray(pnl_r, dtype=np.float64)
    n = len(pnl_r)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))
    sample = pnl_r[idx]                               # (n_iter, n)

    win_rate = (sample > 0).mean(axis=1) * 100.0
    expectancy = sample.mean(axis=1)
    gross_win = np.where(sample > 0, sample, 0.0).sum(axis=1)
    gross_loss = np.where(sample <= 0, -sample, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(gross_loss > 0, gross_win / gross_loss, pf_cap)
    pf = np.minimum(pf, pf_cap)

    def band(a):
        return dict(p5=round(float(np.percentile(a, 5)), 4),
                    p50=round(float(np.percentile(a, 50)), 4),
                    p95=round(float(np.percentile(a, 95)), 4))

    # point estimates on the observed sample
    gw = pnl_r[pnl_r > 0].sum()
    gl = abs(pnl_r[pnl_r <= 0].sum())
    point_pf = min(gw / gl, pf_cap) if gl > 0 else pf_cap
    return dict(
        n_trades=n, n_iter=n_iter,
        profit_factor=dict(point=round(float(point_pf), 4), **band(pf)),
        win_rate=dict(point=round(float((pnl_r > 0).mean() * 100), 4), **band(win_rate)),
        expectancy_r=dict(point=round(float(pnl_r.mean()), 4), **band(expectancy)),
    )


def print_ci(ci, label="pooled trades"):
    if ci is None:
        print(f"  (no trades to bootstrap for {label})")
        return
    print(f"Bootstrap CI ({label}): {ci['n_trades']} trades, {ci['n_iter']} resamples")
    for key, unit in (("profit_factor", ""), ("win_rate", "%"), ("expectancy_r", " R")):
        b = ci[key]
        print(f"  {key:<14} point={b['point']}{unit}   "
              f"90% CI [{b['p5']}{unit} .. {b['p95']}{unit}]  (median {b['p50']}{unit})")


if __name__ == "__main__":
    from wfo.optimize_atr import best_pooled, optimize_pooled
    from wfo.pooling import pooled_backtest, prepare_universe

    print("Preparing universe + finding universal rule (seed=42) ...")
    universe, _ = prepare_universe()
    b = best_pooled(optimize_pooled(universe, n_trials=300))
    rule = {k: b[k] for k in ("tp_atr_mult", "sl_atr_mult",
                              "trail_act_atr_mult", "trail_buf_atr_mult")}
    print(f"Universal rule: " + ", ".join(f"{k}={v:.2f}" for k, v in rule.items()))

    pooled = pooled_backtest(universe, **rule)
    print(f"\nIN-SAMPLE pooled (full history) — Step 6 bootstrap demo "
          f"(OOS bootstrap comes after Step 7):\n")
    print_ci(bootstrap_ci(pooled["pnl_r"]), label="in-sample pooled")
