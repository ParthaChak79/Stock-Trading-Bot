"""
================================================================================
 WEEKLY STOCK SCREENER
================================================================================
Standalone script — independent of app.py's live trading bot process. Queries
TradingView's own screener (the same one shown in the screenshot this was
built from) directly via its scanner API and sends the matches to Telegram.

This is TradingView's actual undocumented scanner endpoint
(scanner.tradingview.com) — not an official public API, but it's what
tradingview.com's own website calls, requires no auth, and every field below
was verified against the live endpoint (see `metainfo` at the same host for
the full field list) rather than guessed. It returns TradingView's own
precomputed values, so results match what you'd see in the screener UI
itself — no independent per-stock computation/approximation needed.

Active filters (from the screenshot's filter chips — chips left as bare
dropdowns with no value, e.g. Watchlist/Index/P.E/Sector/Analyst rating/Beta/
earnings-date/MACD, were not filters and are not applied here):
    Price > EMA(50)              Chg 1M > 1%              Mkt Cap > 10B INR
    EPS growth (TTM YoY) > 20%   Div yield (TTM) > 0%     Perf 5Y < 600%
    Revenue growth (TTM YoY)>20% PEG (TTM) < 1.5           ROE (TTM) > 20%
    Price > 200 INR              Debt/Equity (FQ) < 0.5   Exchange = NSE

The screenshot showed two overlapping "Mkt Cap" filters (>1B and >10B INR);
collapsed here to the stricter >10B threshold.

--------------------------------------------------------------------------------
RUNNING IT EVERY SUNDAY
--------------------------------------------------------------------------------
This script runs once and exits — schedule it with cron (or Task Scheduler /
launchd) rather than building a second always-on process alongside app.py:

    # crontab -e  (runs every Sunday at 8:00 AM local time)
    0 8 * * 0 cd "/path/to/telegram_trading_bot" && /usr/bin/python3 stock_screener.py >> screener.log 2>&1

Fast (a single API call, typically well under a second) — no per-stock
looping is needed since TradingView's scanner does the filtering server-side.
================================================================================
"""

import os
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
IST = timezone(timedelta(hours=5, minutes=30))


def clean_env_var(value):
    if not value:
        return value
    return value.split('#')[0].strip()


load_dotenv(ENV_FILE)
TELEGRAM_BOT_TOKEN = clean_env_var(os.getenv("TELEGRAM_BOT_TOKEN"))
TELEGRAM_CHAT_ID = clean_env_var(os.getenv("TELEGRAM_CHAT_ID"))

# ----------------------------------------------------------------------------
# TradingView scanner
# ----------------------------------------------------------------------------
TV_SCAN_URL = "https://scanner.tradingview.com/india/scan"
TV_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}

# Field names verified against scanner.tradingview.com/india/metainfo.
COLUMNS = [
    "name", "description", "sector", "close", "market_cap_basic",
    "return_on_equity", "price_earnings_growth_ttm", "debt_to_equity_fq",
    "earnings_per_share_diluted_yoy_growth_ttm", "total_revenue_yoy_growth_ttm",
    "dividends_yield_current", "Perf.1M", "Perf.5Y",
]

FILTERS = [
    {"left": "exchange", "operation": "equal", "right": "NSE"},
    {"left": "close", "operation": "greater", "right": 200},                       # Price > 200 INR
    {"left": "market_cap_basic", "operation": "greater", "right": 10_000_000_000},  # Mkt cap > 10B INR
    {"left": "close", "operation": "greater", "right": "EMA50"},                   # Price > EMA, 50
    {"left": "Perf.1M", "operation": "greater", "right": 1},                       # Chg, 1M > 1%
    {"left": "Perf.5Y", "operation": "less", "right": 600},                        # Perf, 5Y < 600%
    {"left": "earnings_per_share_diluted_yoy_growth_ttm", "operation": "greater", "right": 20},  # EPS dil growth, TTM YoY > 20%
    {"left": "total_revenue_yoy_growth_ttm", "operation": "greater", "right": 20},  # Revenue growth, TTM YoY > 20%
    {"left": "dividends_yield_current", "operation": "greater", "right": 0},        # Div yield, TTM > 0%
    {"left": "price_earnings_growth_ttm", "operation": "less", "right": 1.5},       # PEG, TTM < 1.5
    {"left": "return_on_equity", "operation": "greater", "right": 20},              # ROE, TTM > 20%
    {"left": "debt_to_equity_fq", "operation": "less", "right": 0.5},               # Debt/equity, FQ < 0.5
]

RESULT_LIMIT = 150   # rows fetched from TradingView (generous headroom over typical match counts)
DISPLAY_LIMIT = 30   # rows shown in the Telegram message


def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- Telegram credentials missing! Please configure .env file ---")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending telegram message: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def fetch_screener_matches():
    """Query TradingView's scanner for stocks passing all FILTERS. Returns
    (matches, total_count), where matches is a list of column-name -> value
    dicts, or ([], 0) on failure."""
    body = {
        "filter": FILTERS,
        "options": {"lang": "en"},
        "markets": ["india"],
        "symbols": {},
        "columns": COLUMNS,
        "sort": {"sortBy": "return_on_equity", "sortOrder": "desc"},
        "range": [0, RESULT_LIMIT],
    }
    try:
        resp = requests.post(TV_SCAN_URL, headers=TV_HEADERS, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = [dict(zip(COLUMNS, row["d"])) for row in data.get("data", [])]
        return rows, data.get("totalCount", len(rows))
    except Exception as e:
        print(f"[Screener] Error querying TradingView scanner: {e}")
        return [], 0


def format_screener_message(matches, total_count):
    header = (f"🔎 <b>Weekly Stock Screener — {datetime.now(IST).strftime('%d %b %Y')}</b>\n"
              f"{total_count} NSE stocks matched.\n")
    if not matches:
        return header + "\nNo stocks met all criteria this week."

    lines = [header]
    for m in matches[:DISPLAY_LIMIT]:
        lines.append(
            f"\n• <b>{m['description']} ({m['name']})</b>\n"
            f"  {m['sector']}\n"
            f"  💵 ₹{m['close']:.2f} | Mkt Cap ₹{m['market_cap_basic'] / 1e7:,.0f} Cr\n"
            f"  📈 ROE {m['return_on_equity']:.1f}% | PEG {m['price_earnings_growth_ttm']:.2f} | "
            f"Debt/Eq {m['debt_to_equity_fq']:.2f}\n"
            f"  🚀 EPS Growth {m['earnings_per_share_diluted_yoy_growth_ttm']:.1f}% | "
            f"Rev Growth {m['total_revenue_yoy_growth_ttm']:.1f}%"
        )
    if total_count > DISPLAY_LIMIT:
        lines.append(f"\n<i>+ {total_count - DISPLAY_LIMIT} more not shown.</i>")
    return "\n".join(lines)


def run_screener():
    print(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Running weekly stock screener...")
    matches, total_count = fetch_screener_matches()
    print(f"[Screener] {total_count} stocks matched.")
    message = format_screener_message(matches, total_count)
    if send_telegram_message(message):
        print("[Screener] Sent results to Telegram.")
    else:
        print("[Screener] Failed to send results to Telegram.")


if __name__ == "__main__":
    run_screener()
