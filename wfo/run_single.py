"""
Step 2/3 checkpoint: backtest ONE stock with ONE fixed parameter set and print
the results, so they can be sanity-checked against the TradingView backtest
before we build the optimizer on top.

Usage:
    wfo/.venv/bin/python wfo/run_single.py            # defaults to NH, config params
    wfo/.venv/bin/python wfo/run_single.py NH
"""

import sys

from wfo.data import get_ohlcv, load_config
from wfo.strategy import backtest, to_portfolio


def main(symbol: str = "NH"):
    cfg = load_config()
    if symbol not in cfg:
        raise SystemExit(f"{symbol} not found in stocks_config.json")
    entry = cfg[symbol]

    df = get_ohlcv(symbol, entry)
    print(f"=== {symbol} ({entry.get('name', '')}) ===")
    print(f"Data: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()} "
          f"(yfinance: {entry.get('yf_ticker', symbol + '.NS')})\n")

    # Current manually-tuned params from stocks_config.json
    tp, sl = entry["tp"], entry["sl"]
    trail_act, trail_buf = entry["trail_act"], entry["trail_buf"]
    print(f"Current manual params: tp={tp}  sl={sl}  "
          f"trail_act={trail_act}  trail_buf={trail_buf}\n")

    metrics, trades = backtest(df, tp_pct=tp, sl_pct=sl,
                               trail_activation_pct=trail_act,
                               trail_offset_pct=trail_buf)

    print("FAITHFUL ENGINE (validated logic, matches TradingView methodology):")
    print(f"  Trades:        {metrics['num_trades']}")
    print(f"  Win rate:      {metrics['win_rate']}%")
    print(f"  Profit factor: {metrics['profit_factor']}")
    print(f"  Total return:  {metrics['total_return_pct']}%")
    print(f"  Max drawdown:  {metrics['max_drawdown_pct']}%\n")

    # vectorbt cross-check
    pf = to_portfolio(df, trades)
    print("VECTORBT CROSS-CHECK (Portfolio.from_orders):")
    print(f"  Trades:        {pf.trades.count()}")
    wr = pf.trades.win_rate()
    pf_val = pf.trades.profit_factor()
    print(f"  Win rate:      {round(float(wr) * 100, 2)}%")
    print(f"  Profit factor: {round(float(pf_val), 3)}")
    print(f"  Total return:  {round(float(pf.total_return()) * 100, 2)}%")
    print(f"  Max drawdown:  {round(float(pf.max_drawdown()) * 100, 2)}%\n")

    # Show the trades for line-by-line comparison against TradingView's list
    e_idx, x_idx, e_px, x_px, pnl = trades
    print("ALL TRADES (compare dates/prices to your TradingView trade list):")
    print(f"{'#':>3} {'entry_date':>12} {'exit_date':>12} "
          f"{'entry':>9} {'exit':>9} {'pnl%':>7}")
    for j in range(len(pnl)):
        ed = df.index[e_idx[j]].date()
        xd = df.index[x_idx[j]].date()
        print(f"{j + 1:>3} {str(ed):>12} {str(xd):>12} "
              f"{e_px[j]:>9.2f} {x_px[j]:>9.2f} {pnl[j]:>7.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NH")
