"""
================================================================================
 TRADINGVIEW FUNDAMENTALS  —  quantitative data source for the scoring engine
================================================================================
Fetches the quantitative fields the scoring engine (scoring_engine.py) needs
from TradingView's undocumented scanner API (scanner.tradingview.com/india/scan)
— the same host stock_screener.py already uses. No auth. Every field name below
was verified against scanner.tradingview.com/india/metainfo and cross-checked
against live values (BEL / LUPIN / ICICIBANK / MARUTI) before use, per the
project's "verify, never guess scanner fields" rule.

Responsibilities:
  - resolve NSE symbols to TradingView tickers (only BAJAJAUTO needs aliasing;
    M&M and the rest resolve as "NSE:<symbol>" verbatim)
  - batch-fetch fundamentals for a list of symbols -> {norm_key: data dict}
  - compute sector-average P/E and EV/EBITDA across the broad NSE universe
  - apply the spec's hard sector/industry/ticker exclusion filter (Section 8)

Notes on data availability observed live:
  - price-to-book is sparsely populated via the scanner -> often None -> the
    scoring engine defaults price_to_book to 50 (neutral), per spec.
  - ROCE / EBITDA / EV-EBITDA are None for banks -> handled by is_bank + defaults.
================================================================================
"""

import time
import requests

TV_SCAN_URL = "https://scanner.tradingview.com/india/scan"
TV_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}

# NSE symbol -> TradingView symbol, only where they diverge (verified live).
# Everything else resolves as "NSE:<symbol>" unchanged, including "M&M".
TV_SYMBOL_ALIASES = {
    "BAJAJAUTO": "BAJAJ_AUTO",
}

# Scanner columns -> our flat data keys. Order matters (zipped positionally).
COLUMNS = [
    "name", "description", "sector", "industry", "close", "market_cap_basic",
    "return_on_equity", "return_on_capital_employed_fy", "price_earnings_ttm",
    "price_book_fq", "price_book_current", "debt_to_equity_fq",
    "price_52_week_high", "enterprise_value_ebitda_current", "free_cash_flow_fy",
    "total_revenue_yoy_growth_ttm", "total_revenue_cagr_5y",
    "net_income_yoy_growth_ttm", "net_income_cagr_5y", "recommendation_mark",
    "dividends_yield_current",
]

# ---------------------------------------------------------------------------
# Exclusion mapping (spec Section 8), expressed against TradingView taxonomy.
# TradingView's sector/industry labels differ from the spec's shorthand, so the
# spec's EXCLUDED_SECTORS {"IT","Energy","Oil & Gas","FMCG","Insurance"} plus the
# "also check" list (airlines/textiles/paper/real estate/apparel/media) are
# translated here to TradingView industry/sector strings verified live.
# ---------------------------------------------------------------------------
EXCLUDED_TICKERS = {"SBIN", "ADANIPORTS"}  # spec EXCLUDED_TICKERS (PSU bank, ports)

# Whole TradingView sectors that map cleanly to a spec exclusion:
EXCLUDED_SECTORS = {
    "Energy Minerals",      # coal + integrated/refining oil & gas (spec: Energy, Oil & Gas)
    "Consumer Non-Durables",  # FMCG staples / tobacco / beverages (spec: FMCG)
}

# Industry substrings (case-insensitive) that trigger exclusion:
EXCLUDED_INDUSTRY_KEYWORDS = (
    "information technology services",  # spec: IT (legacy outsourcing)
    "insurance",                        # spec: Insurance
    "oil ", "oil refining", "integrated oil", "gas pipelines",  # spec: Oil & Gas
    "coal",                             # spec: Energy (coal mining)
    "airlines",                         # spec "also check": Airlines
    "marine shipping", "water transport",  # ports/shipping (shipbuilders kept, see below)
    "textiles", "apparel", "footwear",  # spec "also check": Textiles / Apparel & Footwear
    "pulp & paper", "paper",            # spec "also check": Pulp & Paper
    "real estate development", "homebuilding",  # spec "also check": Residential Real Estate
    "broadcasting", "movies/entertainment", "cable/satellite",
    "media conglomerates", "publishing",  # spec "also check": Media & Entertainment
)

# Shipbuilders are manufacturing, not ports/shipping — never exclude these even if
# an industry keyword would otherwise catch them (spec Section 8 explicit carve-out).
SHIPBUILDER_KEYWORDS = ("shipbuild", "aerospace & defense", "defense")


def norm_key(symbol: str) -> str:
    """Separator-insensitive matching key so a TradingView `name` and a list
    ticker for the same company collapse together:
      'M&M'->'MM', 'M_M'->'MM', 'BAJAJ_AUTO'->'BAJAJAUTO', 'BAJAJAUTO'->'BAJAJAUTO'."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def tv_ticker(nse_symbol: str) -> str:
    return "NSE:" + TV_SYMBOL_ALIASES.get(nse_symbol.upper(), nse_symbol)


def is_bank(data: dict) -> bool:
    return "banks" in (data.get("industry") or "").lower()


def is_excluded(data: dict) -> bool:
    """Hard sector/industry/ticker exclusion (spec Section 8). data must carry
    'name', 'sector', 'industry'."""
    name = (data.get("name") or "").upper()
    if norm_key(name) in {norm_key(t) for t in EXCLUDED_TICKERS}:
        return True
    industry = (data.get("industry") or "").lower()
    sector = data.get("sector") or ""
    # Shipbuilder / defence carve-out wins over the shipping keyword.
    if any(k in industry for k in SHIPBUILDER_KEYWORDS):
        return False
    if sector in EXCLUDED_SECTORS:
        return True
    if any(k in industry for k in EXCLUDED_INDUSTRY_KEYWORDS):
        return True
    return False


def _post(body):
    resp = requests.post(TV_SCAN_URL, headers=TV_HEADERS, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _row_to_data(row):
    raw = dict(zip(COLUMNS, row["d"]))
    mcap = raw.get("market_cap_basic")
    fcf = raw.get("free_cash_flow_fy")
    fcf_yield = (fcf / mcap * 100) if (fcf is not None and mcap) else None
    mark = raw.get("recommendation_mark")
    # TradingView recommendation_mark: 1.0=Strong Buy ... 5.0=Strong Sell.
    # Map linearly to a percent-buy proxy for score_analyst_revisions.
    buy = max(0.0, min(100.0, (5 - mark) / 4 * 100)) if mark else None
    pb = raw.get("price_book_fq")
    if pb is None:
        pb = raw.get("price_book_current")
    return {
        "name": raw.get("name"),
        "description": raw.get("description"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        "ltp": raw.get("close"),
        "h52": raw.get("price_52_week_high"),
        "mcap": (mcap / 1e7) if mcap else None,   # -> ₹ crores
        "roe": raw.get("return_on_equity"),
        "roce": raw.get("return_on_capital_employed_fy"),
        "pe": raw.get("price_earnings_ttm"),
        "pb": pb,
        "de": raw.get("debt_to_equity_fq"),
        "ev_ebitda": raw.get("enterprise_value_ebitda_current"),
        "fcf_yield": fcf_yield,
        "sg1y": raw.get("total_revenue_yoy_growth_ttm"),
        "sg3y": None,   # scanner exposes only 5y revenue CAGR
        "sg5y": raw.get("total_revenue_cagr_5y"),
        "pg1y": raw.get("net_income_yoy_growth_ttm"),
        "pg3y": None,   # scanner exposes only 5y net-income CAGR
        "pg5y": raw.get("net_income_cagr_5y"),
        "buy": buy,
        "ld": None,     # no reliable listing date in scanner -> longevity uses qualitative prior
    }


def fetch_fundamentals(nse_symbols, batch_size=100):
    """Fetch fundamentals for a list of NSE symbols. Returns {norm_key: data}.
    Symbols the scanner cannot resolve are silently absent from the result and
    logged by the caller (they simply don't get scored)."""
    out = {}
    symbols = list(dict.fromkeys(nse_symbols))  # de-dupe, keep order
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        body = {
            "symbols": {"tickers": [tv_ticker(s) for s in chunk], "query": {"types": []}},
            "columns": COLUMNS,
        }
        try:
            data = _post(body)
        except Exception as e:
            print(f"[tv_fundamentals] batch fetch failed: {e}")
            continue
        for row in data.get("data", []):
            d = _row_to_data(row)
            d["is_bank"] = is_bank(d)
            out[norm_key(d["name"])] = d
        time.sleep(0.3)
    return out


def fetch_sector_averages(min_market_cap=10_000_000_000):
    """Median P/E and EV/EBITDA per TradingView sector across NSE stocks above a
    market-cap floor. Median (not mean) to stay robust to outliers. Returns
    ({sector: avg_pe}, {sector: avg_ev_ebitda}); empty dicts on failure (the
    scoring engine then falls back to neutral valuation defaults)."""
    body = {
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "NSE"},
            {"left": "market_cap_basic", "operation": "greater", "right": min_market_cap},
        ],
        "options": {"lang": "en"},
        "markets": ["india"],
        "columns": ["sector", "price_earnings_ttm", "enterprise_value_ebitda_current"],
        "range": [0, 2000],
    }
    try:
        data = _post(body)
    except Exception as e:
        print(f"[tv_fundamentals] sector averages fetch failed: {e}")
        return {}, {}

    pe_by, ev_by = {}, {}
    for row in data.get("data", []):
        sector, pe, ev = row["d"]
        if not sector:
            continue
        if pe is not None and pe > 0:
            pe_by.setdefault(sector, []).append(pe)
        if ev is not None and ev > 0:
            ev_by.setdefault(sector, []).append(ev)

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)

    return ({s: median(v) for s, v in pe_by.items()},
            {s: median(v) for s, v in ev_by.items()})
