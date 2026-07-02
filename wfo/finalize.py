"""
Final step: assign the OOS-validated universal exit rule to all 45 stocks.

The per-stock personalized-exit layer was REMOVED after testing showed it can't be
trusted from in-sample data: three separate objectives all overfit the ~15-trade
per-stock samples --
  - profit-factor-in-R  -> pushed stops very WIDE (never lose)
  - expectancy-in-R     -> pushed stops TINY (inflate the take-profit's R payoff)
  - mean-%-return       -> picked LOTTERY rules (20-40% win rate, mean carried by a
                           few rare +350..+780% winners)
The common cause is fundamental: optimizing any point metric on ~15 fat-tailed
trend trades overfits to rare winners. Only out-of-sample evidence could justify a
per-stock deviation, and the stocks don't have enough trades for that. So every
stock uses the universal rule -- the one exit validated out-of-sample -- which is
exactly why we pooled in the first place.

This module reports the universal rule, its pooled OOS confidence intervals, and
each stock's trade count + mean %-return under the universal rule (informational).

Outputs: wfo/results/universal_exit_summary.{json,csv}
"""

import json
import os

from wfo.bootstrap import bootstrap_ci
from wfo.pooling import prepare_universe
from wfo.strategy import _simulate_atr
from wfo.optimize_atr import edge_check
from wfo.walkforward_pooled import (consensus_rule, evaluate_fixed_oos,
                                    walk_forward_pooled_atr)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KEYS = ("tp_atr_mult", "sl_atr_mult", "trail_act_atr_mult", "trail_buf_atr_mult")


def _sim(u, rule):
    return _simulate_atr(u["opens"], u["highs"], u["lows"], u["atr"], u["entries"],
                         rule["tp_atr_mult"], rule["sl_atr_mult"],
                         rule["trail_act_atr_mult"], rule["trail_buf_atr_mult"], True)


def per_stock_universal(universe, universal):
    """Every stock uses the universal rule; report its per-stock trade count and
    mean %-return (informational only — no per-stock optimization)."""
    rows = []
    for u in universe:
        p = _sim(u, universal)[4]                 # pnl_pct under the universal rule
        rows.append(dict(symbol=u["symbol"], rule="universal",
                         trades=int(len(p)),
                         mean_pct=round(float(p.mean()), 2) if len(p) else None))
    rows.sort(key=lambda r: -r["trades"])
    return rows


def main(n_trials_wf=150):
    print("Preparing universe ...")
    universe, _ = prepare_universe()

    print("Walk-forward (calendar, pooled) -> window rules -> consensus ...")
    wf = walk_forward_pooled_atr(universe, n_trials=n_trials_wf)
    universal = consensus_rule(wf)
    print("CONSENSUS UNIVERSAL RULE: " + ", ".join(f"{k}={universal[k]}" for k in KEYS))

    # edge-of-grid check on EACH per-window optimization (tp is fixed, so excluded)
    print("Per-window edge-of-grid check (sl/trail only; tp fixed):")
    any_edge = False
    for w in wf["per_window"]:
        if not w["valid"]:
            continue
        flags = edge_check(w["params"])
        if flags:
            any_edge = True
            print(f"  W{w['window']}: " + "; ".join(flags))
    if not any_edge:
        print("  OK - no optimized param hugs a boundary in any window.")

    print("Validating the fixed universal rule out-of-sample ...")
    oos = evaluate_fixed_oos(universe, universal)
    ci = bootstrap_ci(oos["pnl_r"])

    per_stock = per_stock_universal(universe, universal)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = dict(universal_rule=universal, applies_to="all_45_stocks",
               oos_start=oos["oos_start"], oos_trades=int(len(oos["pnl_r"])),
               oos_ci=ci,
               per_window=[{k: r[k] for k in ("window", "valid", "params",
                            "train", "test", "oos_stats")} for r in wf["per_window"]],
               per_stock=per_stock,
               personalized_layer="dropped: in-sample per-stock optimization overfits "
                                  "fat-tailed returns on ~15-trade samples (3 objectives "
                                  "tested); no stock has enough OOS evidence to deviate")
    with open(os.path.join(RESULTS_DIR, "universal_exit_summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    csv_path = os.path.join(RESULTS_DIR, "universal_exit_summary.csv")
    with open(csv_path, "w") as f:
        f.write("symbol,rule,tp_atr_mult,sl_atr_mult,trail_act_atr_mult,"
                "trail_buf_atr_mult,trades_under_universal,mean_pct_under_universal\n")
        for r in per_stock:
            f.write(f"{r['symbol']},universal,{universal['tp_atr_mult']},"
                    f"{universal['sl_atr_mult']},{universal['trail_act_atr_mult']},"
                    f"{universal['trail_buf_atr_mult']},{r['trades']},"
                    f"{'' if r['mean_pct'] is None else r['mean_pct']}\n")

    _report(universal, oos, ci, per_stock)
    return out


def _report(universal, oos, ci, per_stock):
    print("\n" + "=" * 70)
    print("RECOMMENDED UNIVERSAL EXIT RULE (ATR multiples) — APPLIES TO ALL 45 STOCKS:")
    for k in KEYS:
        print(f"  {k:<20} = {universal[k]}")
    print(f"\nOUT-OF-SAMPLE (fixed rule, from {oos['oos_start']}): "
          f"{int(len(oos['pnl_r']))} trades")
    for key, unit in (("profit_factor", ""), ("win_rate", "%"), ("expectancy_r", " R")):
        b = ci[key]
        print(f"  {key:<14} {b['point']}{unit}  90% CI [{b['p5']}{unit} .. {b['p95']}{unit}]")

    n_ge15 = sum(1 for r in per_stock if r["trades"] >= 15)
    print(f"\nPer-stock (all use the universal rule): {len(per_stock)} stocks, "
          f"{n_ge15} with >= 15 own trades under it.")
    print("Personalized-exit layer: DROPPED (in-sample per-stock tuning overfits; "
          "see module docstring).")


if __name__ == "__main__":
    main()
