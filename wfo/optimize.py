"""
Optuna optimization for the AI Wealth Builder strategy (single price series).

Objective: maximize TradingView-style (compounded, currency-based) profit factor,
GUARDED so it can't chase noise:
  - trials with fewer than `min_trades` closed trades are rejected
  - trials with non-positive total return are rejected

Search space uses the manually-explored ranges, with the user's ordering rules
enforced by construction (every sampled trial is valid, so no TPE budget is
wasted on infeasible combos):
  tp_pct               : 0.10 - 0.35
  sl_pct               : within +/-0.15 of tp   (|tp - sl| <= 15%)
  trail_activation_pct : 0.08 - 0.20, strictly below tp
  trail_offset_pct     : 0.03 - 0.12, at least 0.02 below trail_activation
=> guarantees  tp > trail_act > trail_buf  and  trail_buf <= trail_act - 0.02.
"""

import optuna

from wfo.strategy import backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Base ranges (match the manually-explored space)
TP_LO, TP_HI = 0.10, 0.35
SL_LO, SL_HI = 0.08, 0.32
TA_LO, TA_HI = 0.08, 0.20
TB_LO, TB_HI = 0.03, 0.12
TP_SL_MAX_GAP = 0.15         # |tp - sl| <= 15%
TB_BELOW_TA = 0.02           # trail_buf at least 2% below trail_act
PF_CAP = 100.0               # cap inf/huge PF so it doesn't dominate the sampler

METRIC_KEYS = ("num_trades", "win_rate", "profit_factor", "profit_factor_pct",
               "total_return_pct", "max_drawdown_pct")


def _suggest_params(trial):
    """Define-by-run sampling that always satisfies the ordering rules."""
    tp = trial.suggest_float("tp_pct", TP_LO, TP_HI)
    sl = trial.suggest_float("sl_pct",
                             max(SL_LO, tp - TP_SL_MAX_GAP),
                             min(SL_HI, tp + TP_SL_MAX_GAP))
    trail_act = trial.suggest_float("trail_activation_pct",
                                    TA_LO, min(TA_HI, tp - 1e-9))
    trail_buf = trial.suggest_float("trail_offset_pct",
                                    TB_LO, min(TB_HI, trail_act - TB_BELOW_TA))
    return tp, sl, trail_act, trail_buf


def make_objective(df, min_trades=10, entry_params=None):
    entry_params = entry_params or {}

    def objective(trial):
        tp, sl, trail_act, trail_buf = _suggest_params(trial)
        metrics, _ = backtest(df, tp_pct=tp, sl_pct=sl,
                              trail_activation_pct=trail_act,
                              trail_offset_pct=trail_buf, **entry_params)
        # stash metrics for reporting on the best trial
        for k in METRIC_KEYS:
            v = metrics.get(k)
            trial.set_user_attr(k, None if v is None or v != v else float(v))

        # guard: enough trades AND net-positive
        if metrics["num_trades"] < min_trades or not (metrics["total_return_pct"] > 0):
            return -1.0
        pf = metrics["profit_factor"]
        if pf is None or pf != pf or pf == float("inf"):
            pf = PF_CAP
        return min(float(pf), PF_CAP)

    return objective


def optimize(df, n_trials=200, seed=42, min_trades=10, entry_params=None):
    """Run a TPE study; returns the optuna.Study."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(df, min_trades, entry_params),
                   n_trials=n_trials, show_progress_bar=False)
    return study


def best_params(study) -> dict:
    """Best trial's params + its stored metrics as a flat dict.

    `valid` is False when EVERY trial failed the guard (best objective == -1.0),
    i.e. no parameter set produced >= min_trades and a positive return on this
    slice. Callers should treat an invalid result as "no params found", not as a
    real recommendation.
    """
    t = study.best_trial
    out = dict(t.params)
    out.update({k: t.user_attrs.get(k) for k in METRIC_KEYS})
    out["objective_pf"] = t.value
    out["valid"] = t.value is not None and t.value > -1.0
    return out


if __name__ == "__main__":
    import sys
    from wfo.data import get_ohlcv, load_config

    symbol = sys.argv[1] if len(sys.argv) > 1 else "NH"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cfg = load_config()
    entry = cfg[symbol]
    df = get_ohlcv(symbol, entry)
    print(f"=== Optimizing {symbol} ({entry.get('name','')}) on full history ===")
    print(f"Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Trials: {n_trials}  (TPE, seed=42)\n")

    study = optimize(df, n_trials=n_trials)
    bp = best_params(study)

    print("BEST PARAMS (walk-forward not yet applied - full-history fit):")
    print(f"  tp={bp['tp_pct']:.3f}  sl={bp['sl_pct']:.3f}  "
          f"trail_act={bp['trail_activation_pct']:.3f}  trail_buf={bp['trail_offset_pct']:.3f}")
    print(f"  -> trades={int(bp['num_trades'])}  win_rate={bp['win_rate']}%  "
          f"profit_factor={bp['profit_factor']}  return={bp['total_return_pct']}%  "
          f"max_dd={bp['max_drawdown_pct']}%\n")

    print("YOUR CURRENT MANUAL PARAMS (for comparison):")
    from wfo.strategy import backtest as _bt
    m, _ = _bt(df, tp_pct=entry["tp"], sl_pct=entry["sl"],
               trail_activation_pct=entry["trail_act"], trail_offset_pct=entry["trail_buf"])
    print(f"  tp={entry['tp']}  sl={entry['sl']}  "
          f"trail_act={entry['trail_act']}  trail_buf={entry['trail_buf']}")
    print(f"  -> trades={m['num_trades']}  win_rate={m['win_rate']}%  "
          f"profit_factor={m['profit_factor']}  return={m['total_return_pct']}%  "
          f"max_dd={m['max_drawdown_pct']}%")
