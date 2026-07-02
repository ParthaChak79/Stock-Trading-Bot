"""
Step 2 checkpoint: ATR-multiple exit mechanics sanity check on NH.

Shows the OLD %-based trade list next to the NEW ATR-based trade list (with an
illustrative ATR-multiple set), plus a per-trade breakdown proving the exit
levels really are entry +/- (mult * ATR_at_entry) and that R-multiples are clean
(a stop-out is exactly -1.0 R). Entry logic is unchanged.

Usage: wfo/.venv/bin/python -m wfo.run_single_atr [SYMBOL]
"""

import sys

from wfo.data import get_ohlcv, load_config
from wfo.strategy import add_atr, backtest, backtest_atr

# Illustrative ATR multiples (NOT optimized - just to verify mechanics).
# Satisfies the ordering rule: trail_act(4) < tp(6), trail_buf(1) < trail_act(4).
DEMO = dict(tp_atr_mult=6.0, sl_atr_mult=3.0,
            trail_act_atr_mult=4.0, trail_buf_atr_mult=1.0)


def main(symbol="NH"):
    cfg = load_config()
    entry = cfg[symbol]
    df = get_ohlcv(symbol, entry)
    print(f"=== {symbol} ({entry.get('name','')}) — {len(df)} bars "
          f"{df.index[0].date()}..{df.index[-1].date()} ===\n")

    # OLD: %-based exits from stocks_config.json
    old_m, old_tr = backtest(df, tp_pct=entry["tp"], sl_pct=entry["sl"],
                             trail_activation_pct=entry["trail_act"],
                             trail_offset_pct=entry["trail_buf"])
    print(f"OLD %-based (tp={entry['tp']} sl={entry['sl']} "
          f"trail_act={entry['trail_act']} trail_buf={entry['trail_buf']}):")
    print(f"  trades={old_m['num_trades']}  win={old_m['win_rate']}%  "
          f"PF={old_m['profit_factor']}  return={old_m['total_return_pct']}%\n")

    # NEW: ATR-multiple exits (illustrative)
    new_m, new_r, new_tr = backtest_atr(df, **DEMO)
    print(f"NEW ATR-based (tp={DEMO['tp_atr_mult']} sl={DEMO['sl_atr_mult']} "
          f"trail_act={DEMO['trail_act_atr_mult']} trail_buf={DEMO['trail_buf_atr_mult']} x ATR14):")
    print(f"  trades={new_m['num_trades']}  win={new_r['win_rate']}%  "
          f"PF(%)= {new_m['profit_factor']}  PF(R)={new_r['profit_factor_r']}  "
          f"expectancy={new_r['expectancy_r']}R  total={new_r['total_r']}R\n")

    e_idx, x_idx, e_px, x_px, pnl_pct, pnl_r, atr_e = new_tr
    print("NEW trade list (verify: exit ≈ entry ± mult*ATR; stop-outs = -1.00R):")
    print(f"{'#':>3} {'entry_dt':>12} {'exit_dt':>12} {'entry':>9} {'ATR':>7} "
          f"{'tp_lvl':>9} {'sl_lvl':>9} {'exit':>9} {'R':>7} {'kind':>8}")
    for j in range(len(pnl_r)):
        ed = df.index[e_idx[j]].date()
        xd = df.index[x_idx[j]].date()
        tp_lvl = e_px[j] + DEMO["tp_atr_mult"] * atr_e[j]
        sl_lvl = e_px[j] - DEMO["sl_atr_mult"] * atr_e[j]
        kind = ("target" if abs(x_px[j] - tp_lvl) < 1e-6 else
                "stop" if abs(x_px[j] - sl_lvl) < 1e-6 else "trail/be")
        print(f"{j+1:>3} {str(ed):>12} {str(xd):>12} {e_px[j]:>9.2f} {atr_e[j]:>7.2f} "
              f"{tp_lvl:>9.2f} {sl_lvl:>9.2f} {x_px[j]:>9.2f} {pnl_r[j]:>7.2f} {kind:>8}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NH")
