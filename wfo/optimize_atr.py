"""
Step 4: Optuna search for ONE universal ATR-multiple exit rule over the pooled
universe.

Objective: maximize pooled profit factor in R (sum win R / sum |loss R|), guarded
by a pooled min-trade count and positive total R. Ordering rules are enforced by
construction: trail_act < tp and trail_buf < trail_act. An edge-of-grid check
reports any optimum sitting on a range boundary (widen + re-run rather than trust it).
"""

import optuna

from wfo.pooling import pooled_backtest, prepare_universe

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Data-driven ranges (from wfo.ranges), widened on the upper end after the
# first run pinned tp and sl to their ceilings (edge-of-grid check).
RANGES = dict(
    tp_atr_mult=(4.0, 80.0),
    sl_atr_mult=(1.0, 16.0),
    trail_act_atr_mult=(0.5, 25.0),
    trail_buf_atr_mult=(0.25, 3.0),
)
PF_CAP = 100.0
R_KEYS = ("num_trades", "win_rate", "profit_factor_r", "expectancy_r", "total_r")

# Neither magnitude param is optimizable on this trend-selected/survivorship-biased
# universe: every objective rises monotonically with tp (aim higher) and with sl
# (never stop out), so both just run to whatever boundary they're given. They are
# FIXED by domain judgment -- tp at a realistic/achievable target, sl at a
# risk-meaningful stop -- and only the trailing behavior (trail_act/trail_buf),
# which has genuine interior optima, is searched.
FIXED_TP = 20.0
FIXED_SL = 8.0


def _suggest(trial):
    tp = FIXED_TP
    sl = FIXED_SL
    ta = trial.suggest_float("trail_act_atr_mult", RANGES["trail_act_atr_mult"][0],
                             min(RANGES["trail_act_atr_mult"][1], tp - 1e-6))
    tb = trial.suggest_float("trail_buf_atr_mult", RANGES["trail_buf_atr_mult"][0],
                             min(RANGES["trail_buf_atr_mult"][1], ta - 1e-6))
    return tp, sl, ta, tb


def make_pooled_objective(universe, min_pooled=1500):
    def objective(trial):
        tp, sl, ta, tb = _suggest(trial)
        rs = pooled_backtest(universe, tp, sl, ta, tb)["r_stats"]
        for k in R_KEYS:
            v = rs.get(k)
            trial.set_user_attr(k, None if v is None or v != v else float(v))
        if rs["num_trades"] < min_pooled or not (rs["total_r"] > 0):
            return -1.0
        pf = rs["profit_factor_r"]
        if pf is None or pf != pf or pf == float("inf"):
            pf = PF_CAP
        return min(float(pf), PF_CAP)

    return objective


def optimize_pooled(universe, n_trials=300, seed=42, min_pooled=1500):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_pooled_objective(universe, min_pooled), n_trials=n_trials,
                   show_progress_bar=False)
    return study


def best_pooled(study):
    t = study.best_trial
    out = dict(t.params)
    out["tp_atr_mult"] = FIXED_TP          # tp/sl are fixed, not in t.params
    out["sl_atr_mult"] = FIXED_SL
    out.update({k: t.user_attrs.get(k) for k in R_KEYS})
    out["valid"] = t.value is not None and t.value > -1.0
    return out


def edge_check(best, tol=0.02):
    """Flag any OPTIMIZED param sitting within `tol` of a static range boundary.
    tp and sl are fixed (not optimized) so they're excluded."""
    flags = []
    for k, (lo, hi) in RANGES.items():
        if k in ("tp_atr_mult", "sl_atr_mult"):
            continue
        v = best.get(k)
        if v is None:
            continue
        span = hi - lo
        if v <= lo + tol * span:
            flags.append(f"{k}={v:.2f} hugging LOWER edge ({lo}) -> widen down & re-run")
        elif v >= hi - tol * span:
            flags.append(f"{k}={v:.2f} hugging UPPER edge ({hi}) -> widen up & re-run")
    return flags


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    print("Preparing universe ...")
    universe, _ = prepare_universe()
    print(f"Optimizing pooled ATR exit rule over {len(universe)} stocks "
          f"({n_trials} TPE trials) ...\n")

    study = optimize_pooled(universe, n_trials=n_trials)
    b = best_pooled(study)

    print("BEST UNIVERSAL EXIT RULE (pooled, full history):")
    print(f"  tp_atr_mult        = {b['tp_atr_mult']:.2f}")
    print(f"  sl_atr_mult        = {b['sl_atr_mult']:.2f}")
    print(f"  trail_act_atr_mult = {b['trail_act_atr_mult']:.2f}")
    print(f"  trail_buf_atr_mult = {b['trail_buf_atr_mult']:.2f}")
    print(f"\n  pooled trades      = {int(b['num_trades'])}")
    print(f"  profit factor (R)  = {b['profit_factor_r']}")
    print(f"  win rate           = {b['win_rate']}%")
    print(f"  expectancy         = {b['expectancy_r']} R/trade")
    print(f"  total              = {b['total_r']} R")

    flags = edge_check(b)
    print("\nEDGE-OF-GRID CHECK:")
    if flags:
        for f in flags:
            print(f"  !! {f}")
    else:
        print("  OK - no parameter is hugging a range boundary.")
