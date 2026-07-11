"""
================================================================================
 WEEKLY STOCK SCREENER  ->  RANKED LIST BUILDER
================================================================================
Standalone script — independent of app.py's live trading bot process. It does
NOT send anything to Telegram. Its job is to maintain the ranked master list in
top50_stocks_v36.json using the STOCK_SCREENER_SPEC.md scoring engine.

Pipeline (run once, then exits — schedule via cron every Sunday morning):

  1. CANDIDATE DISCOVERY — query TradingView's scanner (the strict fundamental
     screen below) for fresh candidate names. This is only a discovery funnel.
  2. UNION with the tickers already in top50_stocks_v36.json, so existing
     holdings are re-evaluated too (not just new names).
  3. QUANT — batch-fetch fundamentals for the whole union from TradingView
     (tv_fundamentals.py).
  4. EXCLUDE — drop sector/industry/ticker-excluded and stagnant-growth names
     (STOCK_SCREENER_SPEC.md Sections 8-9) before scoring.
  5. QUALITATIVE — Claude (claude_qualitative.py, default claude-opus-4-8 with
     web_search grounding) supplies the researched qualitative criteria, cached
     per ticker so ranks don't jitter on LLM noise.
  6. SCORE — scoring_engine.py combines quant + qualitative into the final score.
  7. REBUILD — keep every stock scoring >= 60, re-rank all of them, and write the
     result back to top50_stocks_v36.json (previous version backed up alongside).

Because step 2 re-scores existing members with fresh data every run, a stock
that decays below 60 falls off the list, and the strict screener in step 1 only
ever *adds* genuinely new qualifying names.

--------------------------------------------------------------------------------
SCHEDULING (every Sunday 8:00 AM local)
--------------------------------------------------------------------------------
    # crontab -e
    0 8 * * 0 cd "/path/to/telegram_trading_bot" && /usr/bin/python3 stock_screener.py >> screener.log 2>&1

Requires ANTHROPIC_API_KEY in .env for the qualitative half. Without it the script
aborts (leaving the list untouched) unless --allow-no-llm is passed.
================================================================================
"""

import os
import sys
import json
import shutil
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

import tv_fundamentals as tvf
import claude_qualitative as gq
from scoring_engine import score_stock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
LIST_FILE = os.path.join(BASE_DIR, "top50_stocks_v36.json")
IST = timezone(timedelta(hours=5, minutes=30))

load_dotenv(ENV_FILE)

# ----------------------------------------------------------------------------
# Candidate-discovery screen (TradingView scanner) — same filters as before.
# ----------------------------------------------------------------------------
TV_SCAN_URL = "https://scanner.tradingview.com/india/scan"
TV_HEADERS = tvf.TV_HEADERS

FILTERS = [
    {"left": "exchange", "operation": "equal", "right": "NSE"},
    {"left": "close", "operation": "greater", "right": 200},                        # Price > 200 INR
    {"left": "market_cap_basic", "operation": "greater", "right": 10_000_000_000},   # Mkt cap > 10B INR
    {"left": "close", "operation": "greater", "right": "EMA50"},                     # Price > EMA, 50
    {"left": "Perf.1M", "operation": "greater", "right": 1},                         # Chg, 1M > 1%
    {"left": "Perf.5Y", "operation": "less", "right": 600},                          # Perf, 5Y < 600%
    {"left": "earnings_per_share_diluted_yoy_growth_ttm", "operation": "greater", "right": 20},  # EPS growth > 20%
    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 20},   # Revenue growth > 20%
    {"left": "dividends_yield_current", "operation": "greater", "right": 0},         # Div yield > 0%
    {"left": "price_earnings_growth_ttm", "operation": "less", "right": 1.5},        # PEG < 1.5
    {"left": "return_on_equity", "operation": "greater", "right": 20},               # ROE > 20%
    {"left": "debt_to_equity_fq", "operation": "less", "right": 0.5},                # Debt/equity < 0.5
]

INCLUSION_THRESHOLD = 60.0
RESULT_LIMIT = 200


def fetch_screener_candidates():
    """Query TradingView's scanner for stocks passing all FILTERS. Returns a list
    of NSE symbols (TradingView `name`), or [] on failure."""
    body = {
        "filter": FILTERS,
        "options": {"lang": "en"},
        "markets": ["india"],
        "symbols": {},
        "columns": ["name"],
        "sort": {"sortBy": "return_on_equity", "sortOrder": "desc"},
        "range": [0, RESULT_LIMIT],
    }
    try:
        resp = requests.post(TV_SCAN_URL, headers=TV_HEADERS, json=body, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        return [row["d"][0] for row in rows]
    except Exception as e:
        print(f"[screener] scanner query failed: {e}")
        return []


def load_existing_list():
    try:
        with open(LIST_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"stocks": []}


def build_ranked_list(allow_no_llm=False):
    now_ist = datetime.now(IST)
    print(f"[{now_ist.strftime('%Y-%m-%d %H:%M:%S')}] Rebuilding ranked list...")

    api_key = gq.load_api_key()
    if not api_key and not allow_no_llm:
        print("[screener] ABORT: ANTHROPIC_API_KEY not set. Rebuilding without the "
              "qualitative half would collapse the list to a handful of names and "
              "overwrite the curated file. Set ANTHROPIC_API_KEY in .env, or pass "
              "--allow-no-llm to intentionally rebuild on quant + neutral "
              "defaults only. Existing list left untouched.")
        return None
    if not api_key:
        print("[screener] WARNING: ANTHROPIC_API_KEY not set — qualitative criteria "
              "use spec defaults (--allow-no-llm). Results will be quant-only.")

    existing = load_existing_list()
    existing_syms = [s["ticker"] for s in existing.get("stocks", [])]
    candidates = fetch_screener_candidates()
    print(f"[screener] {len(candidates)} scanner candidates, "
          f"{len(existing_syms)} existing holdings.")

    # Union (separator-insensitive de-dupe happens inside fetch_fundamentals).
    universe = list(dict.fromkeys(existing_syms + candidates))
    print(f"[screener] scoring universe: {len(universe)} names.")

    quant = tvf.fetch_fundamentals(universe)
    unresolved = [s for s in universe if tvf.norm_key(s) not in quant]
    if unresolved:
        print(f"[screener] {len(unresolved)} names unresolved on TradingView: {unresolved}")

    sector_avg_pe, sector_avg_ev = tvf.fetch_sector_averages()
    print(f"[screener] sector averages computed for {len(sector_avg_pe)} sectors.")

    cache = gq.load_cache()

    results, excluded, stagnant = [], 0, 0
    for key, data in quant.items():
        sector_excluded = tvf.is_excluded(data)
        if sector_excluded:
            excluded += 1
            continue  # never scored (spec Section 8) — also saves an LLM call

        qualitative = gq.get_qualitative(key, data, cache, api_key)
        scored = score_stock(
            ticker=data.get("name") or key,
            data=data,
            qualitative=qualitative,
            sector_avg_pe=sector_avg_pe.get(data.get("sector")),
            sector_avg_ev_ebitda=sector_avg_ev.get(data.get("sector")),
            sector_excluded=sector_excluded,
        )
        if scored["flags"]["stagnant_growth"]:
            stagnant += 1
        if scored["include_in_list"]:
            results.append(scored)

    gq.save_cache(cache)

    results.sort(key=lambda r: r["total"], reverse=True)
    stocks = []
    for rank, r in enumerate(results, start=1):
        stocks.append({
            "rank": rank,
            "ticker": r["ticker"],
            "name": r["name"],
            "sector": r["sector"],
            "total": r["total"],
            "catA": r["catA"],
            "catB": r["catB"],
            "catC": r["catC"],
            "catD": r["catD"],
            "catF": r["catF"],
            "ai_score": r["ai_score"],
            "ltp": r["ltp"],
        })

    output = {
        "as_of_date": now_ist.strftime("%Y-%m-%d"),
        "methodology": (existing.get("methodology") or
                        "v36 - 30-parameter fundamental screener per STOCK_SCREENER_SPEC.md. "
                        "Inclusion threshold: score >= 60."),
        "total_universe_size": len(universe),
        "eligible_after_exclusions": len(universe) - excluded,
        "qualifying_score_60_plus": len(stocks),
        "stocks": stocks,
    }

    if os.path.exists(LIST_FILE):
        shutil.copyfile(LIST_FILE, LIST_FILE + ".bak")
    with open(LIST_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[screener] {excluded} excluded, {stagnant} stagnant, "
          f"{len(stocks)} qualifying (>= {INCLUSION_THRESHOLD}). "
          f"Wrote {os.path.basename(LIST_FILE)} (previous -> .bak).")
    return output


if __name__ == "__main__":
    build_ranked_list(allow_no_llm="--allow-no-llm" in sys.argv)
