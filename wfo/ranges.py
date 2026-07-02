"""
Data-driven ATR-multiple range selection.

Before optimizing, look at what actually happens after entry signals: for every
realistic entry across the universe, measure the Maximum Favorable Excursion
(MFE) and Maximum Adverse Excursion (MAE) in ATR units over a fixed forward
horizon (exit-agnostic, so exit choices don't bias the ranges). The distributions
of MFE/MAE ground the search ranges:

  sl_atr_mult  <- MAE distribution (how far price goes against you before recovering)
  tp_atr_mult  <- MFE distribution (how far price runs in your favor)
  trail_act    <- lower-middle MFE (a favorable move worth locking in), below tp
  trail_buf    <- small, below trail_act

Entries are taken from the illustrative baseline run (realistic, when-flat,
non-overlapping); excursions are measured over a FIXED horizon from each entry so
they aren't truncated by the baseline's own exits.
"""

import numpy as np

from wfo.pooling import DEMO, prepare_universe
from wfo.strategy import _simulate_atr

HORIZON = 250          # ~1 trading year forward per entry


def excursions(universe, horizon=HORIZON):
    mfe, mae = [], []
    for u in universe:
        tr = _simulate_atr(u["opens"], u["highs"], u["lows"], u["atr"],
                           u["entries"], DEMO["tp_atr_mult"], DEMO["sl_atr_mult"],
                           DEMO["trail_act_atr_mult"], DEMO["trail_buf_atr_mult"], True)
        e_idx, atr_e = tr[0], tr[6]
        highs, lows, opens = u["highs"], u["lows"], u["opens"]
        n = len(opens)
        for k in range(len(e_idx)):
            j = e_idx[k]
            a = atr_e[k]
            if a <= 0:
                continue
            end = min(j + horizon, n)
            ep = opens[j]
            mfe.append((highs[j:end].max() - ep) / a)
            mae.append((ep - lows[j:end].min()) / a)
    return np.array(mfe), np.array(mae)


def pct(a, p):
    return round(float(np.percentile(a, p)), 2)


def propose_ranges(mfe, mae):
    # sl: 1 ATR (whipsaw floor) up through the 75th pct of adverse excursion
    sl_lo, sl_hi = 1.0, round(pct(mae, 75), 1)
    # tp: floor well BELOW median MFE so small-target strategies stay in-scope;
    #     ceiling at the 90th pct (fat right tail of trend runs). Edge-of-grid
    #     check widens the top if the optimum pushes against it.
    tp_lo, tp_hi = 4.0, round(pct(mfe, 90), 1)
    # trail_act: modest favorable move up to ~60th pct MFE, kept below tp
    ta_lo, ta_hi = 2.0, round(pct(mfe, 60), 1)
    # trail_buf: small breakeven+ buffer, below trail_act
    tb_lo, tb_hi = 0.25, 3.0
    return dict(tp_atr_mult=(tp_lo, tp_hi), sl_atr_mult=(sl_lo, sl_hi),
                trail_act_atr_mult=(ta_lo, ta_hi), trail_buf_atr_mult=(tb_lo, tb_hi))


if __name__ == "__main__":
    print("Preparing universe ...")
    universe, _ = prepare_universe()
    mfe, mae = excursions(universe)
    print(f"Measured excursions from {len(mfe)} entries "
          f"(horizon {HORIZON} bars, ATR units)\n")

    print(f"{'pctile':>8}{'MFE (favorable)':>18}{'MAE (adverse)':>16}")
    for p in (10, 25, 50, 60, 70, 75, 90, 95):
        print(f"{p:>7}%{pct(mfe, p):>18}{pct(mae, p):>16}")

    ranges = propose_ranges(mfe, mae)
    print("\nPROPOSED SEARCH RANGES (ATR multiples):")
    for k, (lo, hi) in ranges.items():
        print(f"  {k:<22} {lo} .. {hi}")
