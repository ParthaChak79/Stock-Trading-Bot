"""
Data layer for the AI Wealth Builder walk-forward optimizer.

Pulls daily OHLCV via yfinance using each stock's `yf_ticker` field (falling
back to SYMBOL.NS when absent), and caches to local parquet so repeated runs
don't re-download. All stock metadata is read from the project's
stocks_config.json (symbol -> {name, yf_ticker, current tp/sl/trail params}).
"""

import json
import os

import pandas as pd
import yfinance as yf

# Resolve paths relative to the project root (parent of this wfo/ package)
WFO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(WFO_DIR)
CONFIG_PATH = os.path.join(PROJECT_DIR, "stocks_config.json")
CACHE_DIR = os.path.join(WFO_DIR, "data_cache")

os.makedirs(CACHE_DIR, exist_ok=True)


def load_config() -> dict:
    """Return the full stocks_config.json as {SYMBOL: {...}}."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def resolve_yf_ticker(symbol: str, cfg_entry: dict) -> str:
    """yf_ticker if present, else the SYMBOL.NS fallback."""
    tkr = (cfg_entry or {}).get("yf_ticker")
    return tkr if tkr else f"{symbol}.NS"


def _cache_path(yf_ticker: str, interval: str) -> str:
    safe = yf_ticker.replace(".", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{interval}.parquet")


def get_ohlcv(symbol: str, cfg_entry: dict = None, interval: str = "1d",
              start: str = "2000-01-01", force_refresh: bool = False) -> pd.DataFrame:
    """
    Return a daily OHLCV DataFrame (lowercase columns: open/high/low/close/volume)
    for one stock, using the parquet cache when available.

    Parameters mirror the yfinance download; `force_refresh=True` bypasses cache.
    """
    if cfg_entry is None:
        cfg_entry = load_config().get(symbol, {})
    yf_ticker = resolve_yf_ticker(symbol, cfg_entry)
    path = _cache_path(yf_ticker, interval)

    if os.path.exists(path) and not force_refresh:
        df = pd.read_parquet(path)
        return df

    raw = yf.download(yf_ticker, start=start, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no data for {symbol} (ticker={yf_ticker}). "
            "Check the yf_ticker in stocks_config.json.")

    # Flatten the occasional MultiIndex columns yfinance returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw = raw[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"])
    raw.index = pd.to_datetime(raw.index)
    raw.to_parquet(path)
    return raw


if __name__ == "__main__":
    # Smoke test: load NH and report what we got.
    cfg = load_config()
    entry = cfg["NH"]
    print(f"NH yf_ticker -> {resolve_yf_ticker('NH', entry)}")
    df = get_ohlcv("NH", entry)
    print(f"Loaded {len(df)} bars: {df.index[0].date()} -> {df.index[-1].date()}")
    print(df.tail(3))
