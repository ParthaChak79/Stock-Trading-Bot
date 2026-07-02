"""
Step 7: calendar-time walk-forward on the pooled universe.

Windows are defined in CALENDAR TIME over the universe's date span: the tail
(1 - train_frac) is split into `n_windows` non-overlapping TEST blocks, each
preceded by a rolling TRAIN window. For each window we optimize the universal
ATR exit rule on every stock's trades ENTERED within the train dates (pooled),
then apply that rule to trades entered within the test dates (pooled OOS).

Trades are always simulated over each stock's full price series and then filtered
by entry date, so cutting windows never creates price-series boundary artifacts.
Every window's OOS trades are concatenated into one pooled OOS record for the
Step-6 bootstrap. OOS entry dates + symbols are retained for the Step-8 per-stock
breakdown.
"""

import numpy as np
import optuna
import pandas as pd

from wfo.optimize_atr import FIXED_SL, FIXED_TP, PF_CAP, _suggest
from wfo.pooling import pooled_backtest, prepare_universe
from wfo.strategy import r_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_calendar_windows(dmin, dmax, n_windows=4, train_frac=0.70):
    dmin, dmax = pd.Timestamp(dmin), pd.Timestamp(dmax)
    span = dmax - dmin
    test_region_start = dmin + span * train_frac
    train_len = test_region_start - dmin
    test_size = (dmax - test_region_start) / n_windows
    windows = []
    for i in range(n_windows):
        test_start = test_region_start + test_size * i
        test_end = dmax if i == n_windows - 1 else test_region_start + test_size * (i + 1)
        train_start = max(dmin, test_start - train_len)
        windows.append((train_start, test_start, test_end))
    return windows


def _mask(entry_dates, lo, hi):
    return (entry_dates >= np.datetime64(lo)) & (entry_dates < np.datetime64(hi))


def _optimize_window(universe, lo, hi, n_trials, min_pooled, seed=42):
    def objective(trial):
        tp, sl, ta, tb = _suggest(trial)
        pooled = pooled_backtest(universe, tp, sl, ta, tb)
        pr = pooled["pnl_r"][_mask(pooled["entry_dates"], lo, hi)]
        rs = r_metrics(pr)
        if rs["num_trades"] < min_pooled or not (rs["total_r"] > 0):
            return -1.0
        pf = rs["profit_factor_r"]
        if pf is None or pf != pf or pf == float("inf"):
            pf = PF_CAP
        return min(float(pf), PF_CAP)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    t = study.best_trial
    params = {"tp_atr_mult": FIXED_TP,          # tp/sl fixed, not in t.params
              "sl_atr_mult": FIXED_SL,
              "trail_act_atr_mult": t.params["trail_act_atr_mult"],
              "trail_buf_atr_mult": t.params["trail_buf_atr_mult"]}
    return dict(params=params, valid=t.value is not None and t.value > -1.0)


def walk_forward_pooled_atr(universe, n_windows=4, train_frac=0.70,
                            n_trials=150, min_pooled_window=300, seed=42):
    dmin = min(pd.Timestamp(u["dates"][0]) for u in universe)
    dmax = max(pd.Timestamp(u["dates"][-1]) for u in universe)
    windows = make_calendar_windows(dmin, dmax, n_windows, train_frac)

    per_window = []
    oos_r, oos_dates, oos_syms = [], [], []
    for wi, (tr_start, tr_end, te_end) in enumerate(windows):
        best = _optimize_window(universe, tr_start, tr_end, n_trials,
                                min_pooled_window, seed)
        rec = dict(window=wi, valid=best["valid"], params=best["params"],
                   train=(str(tr_start.date()), str(tr_end.date())),
                   test=(str(tr_end.date()), str(te_end.date())))
        if best["valid"]:
            pooled = pooled_backtest(universe, **best["params"])
            is_mask = _mask(pooled["entry_dates"], tr_start, tr_end)
            oos_mask = _mask(pooled["entry_dates"], tr_end, te_end)
            rec["is_stats"] = r_metrics(pooled["pnl_r"][is_mask])
            rec["oos_stats"] = r_metrics(pooled["pnl_r"][oos_mask])
            oos_r.append(pooled["pnl_r"][oos_mask])
            oos_dates.append(pooled["entry_dates"][oos_mask])
            oos_syms.append(pooled["symbols"][oos_mask])
        else:
            rec["is_stats"] = rec["oos_stats"] = None
        per_window.append(rec)

    pooled_oos = dict(
        pnl_r=np.concatenate(oos_r) if oos_r else np.array([]),
        entry_dates=np.concatenate(oos_dates) if oos_dates else np.array([], dtype="datetime64[ns]"),
        symbols=np.concatenate(oos_syms) if oos_syms else np.array([], dtype=object),
    )
    return dict(per_window=per_window, pooled_oos=pooled_oos)


def consensus_rule(wf):
    """Median of the valid per-window rules -> the single recommended rule."""
    keys = ("tp_atr_mult", "sl_atr_mult", "trail_act_atr_mult", "trail_buf_atr_mult")
    valid = [r["params"] for r in wf["per_window"] if r["valid"]]
    return {k: round(float(np.median([p[k] for p in valid])), 3) for k in keys}


def evaluate_fixed_oos(universe, rule, n_windows=4, train_frac=0.70):
    """Apply ONE fixed rule and collect its trades in the OOS (test) region.

    Unlike walk_forward_pooled_atr (which re-optimizes per window), this measures
    the single recommended rule's out-of-sample behavior as-deployed.
    """
    dmin = min(pd.Timestamp(u["dates"][0]) for u in universe)
    dmax = max(pd.Timestamp(u["dates"][-1]) for u in universe)
    windows = make_calendar_windows(dmin, dmax, n_windows, train_frac)
    oos_lo = windows[0][1]                 # first test_start = start of OOS region
    pooled = pooled_backtest(universe, **rule)
    m = _mask(pooled["entry_dates"], oos_lo, dmax + pd.Timedelta(days=1))
    return dict(pnl_r=pooled["pnl_r"][m], entry_dates=pooled["entry_dates"][m],
                symbols=pooled["symbols"][m], oos_start=str(pd.Timestamp(oos_lo).date()))


if __name__ == "__main__":
    from wfo.bootstrap import bootstrap_ci, print_ci

    print("Preparing universe ...")
    universe, _ = prepare_universe()
    print("Running calendar-time pooled walk-forward (4 windows, 70/30) ...\n")
    wf = walk_forward_pooled_atr(universe, n_trials=150)

    for r in wf["per_window"]:
        print(f"Window {r['window']}: train {r['train'][0]}..{r['train'][1]}  |  "
              f"test {r['test'][0]}..{r['test'][1]}")
        if not r["valid"]:
            print("  NO VALID RULE on train slice\n")
            continue
        p = r["params"]
        print(f"  rule: tp={p['tp_atr_mult']:.1f} sl={p['sl_atr_mult']:.1f} "
              f"trail_act={p['trail_act_atr_mult']:.2f} trail_buf={p['trail_buf_atr_mult']:.2f}")
        for lbl, s in (("IS ", r["is_stats"]), ("OOS", r["oos_stats"])):
            print(f"  {lbl}: trades={s['num_trades']}  PF(R)={s['profit_factor_r']}  "
                  f"win={s['win_rate']}%  exp={s['expectancy_r']}R")
        print()

    oos = wf["pooled_oos"]
    print(f"POOLED OUT-OF-SAMPLE (all test blocks combined): {len(oos['pnl_r'])} trades\n")
    print_ci(bootstrap_ci(oos["pnl_r"]), label="pooled OOS")
