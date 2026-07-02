"""
Step 7 + 8: scale the walk-forward diagnostics across all ~45 stocks (parallel)
and produce the final comparison report.

For each stock:
  - run the full walk-forward + overfitting diagnostics
  - recommended params = the full-history Optuna optimum (what optimization would
    hand you); the walk-forward grade + OOS profit factor tell you how much to
    TRUST that recommendation
  - compare against the current manually-tuned params from stocks_config.json

Outputs (to wfo/results/):
  - wfo_summary.csv   : one row per stock, recommended vs manual + stability
  - wfo_summary.json  : same, with the per-window detail and reasons
"""

import concurrent.futures as cf
import json
import os
import sys

from wfo.data import get_ohlcv, load_config
from wfo.diagnostics import RESULTS_DIR, diagnose
from wfo.strategy import backtest


def diagnose_one(symbol, n_trials=100, n_windows=4, min_trades=10):
    """Worker: full diagnostics for one stock -> compact summary dict."""
    cfg = load_config()
    entry = cfg[symbol]
    df = get_ohlcv(symbol, entry)
    diag = diagnose(symbol, df, n_windows=n_windows, n_trials=n_trials,
                    min_trades=min_trades, save=True)
    full = diag["full_history_best"]
    gap = diag["gap"]
    heat = diag["heatmap"]
    pooled = diag["pooled"]
    pr = diag["plateau_rec"]     # robust plateau recommendation (may be None)
    v = diag["verdict"]          # pooled verdict (primary)

    # current manual params, scored on full history for a like-for-like compare
    man, _ = backtest(df, tp_pct=entry["tp"], sl_pct=entry["sl"],
                      trail_activation_pct=entry["trail_act"],
                      trail_offset_pct=entry["trail_buf"])

    # recommended params = plateau center (robust); fall back to raw peak if absent
    if pr is not None:
        rec_tp, rec_sl = pr["rec_tp"], pr["rec_sl"]
        rec_ta, rec_tb = pr["rec_trail_act"], pr["rec_trail_buf"]
        rec_pf = pr["full_metrics"]["profit_factor"]
        rec_trades = pr["full_metrics"]["num_trades"]
        rec_support = pr["local_support_frac"]
    else:
        rec_tp, rec_sl = round(full["tp_pct"], 3), round(full["sl_pct"], 3)
        rec_ta, rec_tb = round(full["trail_activation_pct"], 3), round(full["trail_offset_pct"], 3)
        rec_pf, rec_trades, rec_support = full["profit_factor"], full["num_trades"], None

    return {
        "symbol": symbol,
        "name": entry.get("name", ""),
        "n_bars": len(df),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "grade": v["grade"],
        "reasons": v["reasons"],
        "window_grade": diag["verdict_window"]["grade"],
        # recommended (plateau center, robust; trust gated by grade)
        "rec_tp": rec_tp,
        "rec_sl": rec_sl,
        "rec_trail_act": rec_ta,
        "rec_trail_buf": rec_tb,
        "rec_full_pf": rec_pf,
        "rec_full_trades": rec_trades,
        "rec_local_support": rec_support,
        # raw peak, for reference / comparison
        "peak_tp": round(full["tp_pct"], 3),
        "peak_sl": round(full["sl_pct"], 3),
        "peak_pf_full": full["profit_factor"],
        # pooled out-of-sample trust signals (primary)
        "pooled_oos_pf": pooled["pooled_oos_pf"],
        "pooled_oos_trades": pooled["pooled_trades"],
        "pooled_win_rate": pooled["pooled_win_rate"],
        "pooled_return_pct": pooled["pooled_return_pct"],
        # per-window walk-forward signals (secondary)
        "oos_pf_median": gap["oos_pf_median"],
        "oos_is_ratio_median": gap["median_oos_is_ratio"],
        "n_valid_windows": gap["n_windows"] - gap["n_invalid"],
        "n_low_oos_windows": gap["n_low_oos"],
        "plateau_frac": heat["plateau_frac"],
        "peak_pf": heat["peak_pf"],
        # manual comparison
        "man_tp": entry["tp"],
        "man_sl": entry["sl"],
        "man_trail_act": entry["trail_act"],
        "man_trail_buf": entry["trail_buf"],
        "man_full_pf": man["profit_factor"],
        "man_full_trades": man["num_trades"],
        "man_win_rate": man["win_rate"],
        "heatmap_png": heat["png_path"],
        "per_window": gap["per_window"],
    }


def run_batch(symbols=None, n_trials=100, max_workers=None):
    cfg = load_config()
    symbols = symbols or list(cfg.keys())
    if max_workers is None:
        max_workers = max(1, min(8, (os.cpu_count() or 4) - 2))

    results, errors = [], []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(diagnose_one, s, n_trials): s for s in symbols}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            try:
                results.append(fut.result())
                print(f"  done: {s}")
            except Exception as exc:  # noqa: BLE001 - one bad stock shouldn't kill the batch
                errors.append((s, str(exc)))
                print(f"  FAILED: {s} -> {exc}")

    grade_rank = {"STABLE": 0, "BORDERLINE": 1, "UNSTABLE": 2}

    def _pf(r):
        pf = r.get("pooled_oos_pf")
        return 0 if pf is None else (1e9 if pf == float("inf") else pf)

    results.sort(key=lambda r: (grade_rank.get(r["grade"], 3), -_pf(r)))
    return results, errors


def write_report(results, errors):
    json_path = os.path.join(RESULTS_DIR, "wfo_summary.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2, default=str)

    csv_path = os.path.join(RESULTS_DIR, "wfo_summary.csv")
    cols = ["symbol", "name", "grade", "pooled_oos_pf", "pooled_oos_trades",
            "pooled_win_rate", "pooled_return_pct", "window_grade",
            "oos_pf_median", "oos_is_ratio_median",
            "n_valid_windows", "n_low_oos_windows", "plateau_frac",
            "rec_tp", "rec_sl", "rec_trail_act", "rec_trail_buf",
            "rec_full_pf", "rec_full_trades", "rec_local_support",
            "peak_tp", "peak_sl", "peak_pf_full",
            "man_tp", "man_sl", "man_trail_act", "man_trail_buf",
            "man_full_pf", "man_full_trades", "man_win_rate",
            "n_bars", "start", "end", "reasons"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            row = []
            for c in cols:
                v = r.get(c)
                if c == "reasons":
                    v = "; ".join(v) if v else ""
                    v = '"' + v.replace('"', "'") + '"'
                row.append("" if v is None else str(v))
            f.write(",".join(row) + "\n")
    return csv_path, json_path


def print_summary(results, errors):
    print(f"\n{'symbol':<12}{'grade':<11}{'poolPF':>7}{'poolN':>6}{'poolWin':>8}"
          f"  {'rec(tp/sl/ta/tb)':<26}{'manPF':>7}")
    for r in results:
        rec = f"{r['rec_tp']}/{r['rec_sl']}/{r['rec_trail_act']}/{r['rec_trail_buf']}"
        pf = r["pooled_oos_pf"]
        pf_s = "-" if pf is None else ("inf" if pf == float("inf") else f"{pf:.2f}")
        win = r["pooled_win_rate"]
        win_s = "-" if win is None or win != win else f"{win:.0f}%"
        print(f"{r['symbol']:<12}{r['grade']:<11}{pf_s:>7}{r['pooled_oos_trades']:>6}"
              f"{win_s:>8}  {rec:<26}{str(r['man_full_pf']):>7}")
    n = len(results)
    from collections import Counter
    counts = Counter(r["grade"] for r in results)
    print(f"\n{n} stocks: " + ", ".join(f"{g}={counts.get(g,0)}"
          for g in ("STABLE", "BORDERLINE", "UNSTABLE")))
    if errors:
        print(f"Errors ({len(errors)}): " + ", ".join(s for s, _ in errors))


if __name__ == "__main__":
    # Optional args: comma-separated symbols, then n_trials
    arg_syms = None
    n_trials = 100
    if len(sys.argv) > 1 and sys.argv[1] != "ALL":
        arg_syms = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        n_trials = int(sys.argv[2])

    print(f"Running batch (n_trials={n_trials}) ...")
    results, errors = run_batch(symbols=arg_syms, n_trials=n_trials)
    csv_path, json_path = write_report(results, errors)
    print_summary(results, errors)
    print(f"\nSaved: {csv_path}\n       {json_path}")
