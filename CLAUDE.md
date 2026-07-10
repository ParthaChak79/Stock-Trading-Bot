# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Telegram Trading Bot — AI Wealth Builder Strategy**

A Python application that monitors NSE (National Stock Exchange) stocks and sends automated BUY/SELL trading alerts via Telegram using the **AI Wealth Builder Strategy**. The bot uses MACD (Moving Average Convergence Divergence) with cooldown mechanics and SMA 50 trend filtering for entry signals, paired with customizable trailing stop configurations per stock for exits. The canonical strategy definition lives in `MACD+SMA.pinescript` (the original TradingView Pine Script); `app.py`'s `analyze_stocks()` and `strategy_optimizer.py`'s backtester are both hand-translated ports of that same logic — if the strategy rules ever need to change, treat the Pine Script as the source of truth and update all three in lockstep.

Key features:
- Real-time stock monitoring with hourly analysis (`analyze_stocks`)
- MACD-based entry signals with cooldown and trend filtering
- Per-stock trailing stop and take-profit configuration
- News sentiment analysis (FinBERT via HuggingFace, with keyword fallback) attached to buy/sell alerts
- Breaking news alerts, independent of the trading signals (`check_news_stream`)
- Earnings-surprise alerts: queries TradingView's scanner for each stock's latest reported quarter (actual/estimate/surprise% for EPS and Revenue) — replaced an earlier NSE-PDF-parsing approach that was unreliable across varying filing layouts (`check_earnings_surprises`)
- Market-crash alert: fires once per day if NIFTY 50 falls ≥2% intraday (`check_market_crash`)
- Daily structured pre-market brief: GIFT Nifty setup (text-mined from news, no free API exists for it), global cues, commodities, FII/DII flows with a computed buy/sell streak, India VIX, prior close + Nifty-50 breadth, overall sentiment (`send_pre_market_report`)
- Weekly portfolio/closed-trades report every Friday (`send_holdings_report`)
- Portfolio state persistence across restarts (saved in `portfolio_state.json`)
- Closed trade history tracking (saved in `closed_trades.json`)

## Architecture & Key Components

### Data Flow
1. **Market Data Ingestion** → `tvdatafeed` (TradingView data) primary, `yfinance` fallback/secondary for indices, commodities, FX, and news
2. **Signal Generation** → MACD/SMA calculations in `app.py` (`calculate_indicators`, `analyze_stocks`)
3. **Position Management** → Entry/exit logic with trailing stops, one position per stock
4. **Notifications** → Telegram (primary) + WhatsApp via Twilio (optional mirror) — all alerts funnel through `send_telegram_message`, which also calls `send_whatsapp_message`
5. **State Persistence** → JSON files track open/closed trades, dedup caches, and alert history across restarts

### Everything is one process, gated by a single `schedule` loop
`app.py` has no web server, queue, or database — `run_scheduler()` at the bottom of the file registers every recurring job against the `schedule` library and then blocks in a `while True: schedule.run_pending()` loop, polling every 60s. All the features above (trading signals, news, earnings, crash alerts, pre-market brief, weekly report) are independent jobs on this one loop, each wrapped in its own try/except so one feature failing (e.g. NSE blocking a request) never takes down the others. When adding a new recurring feature, follow this pattern: write a standalone `check_*`/`send_*` function that swallows its own exceptions, then register it with `schedule.every(...).do(...)` in `run_scheduler()`.

### Core Files

| File | Purpose |
|------|---------|
| `app.py` | Main bot engine; contains all strategy logic, state management, and notification dispatch |
| `strategy_optimizer.py` | Backtest harness for grid-searching optimal exit parameters (TP%, SL%, trail activation%, trail buffer%) for a single ticker at a time |
| `MACD+SMA.pinescript` | Canonical TradingView Pine Script strategy definition ("AI Wealth Builder") — `app.py` and `strategy_optimizer.py` both port this logic |
| `stocks_config.json` | Per-stock configuration: take-profit (tp), stop-loss (sl), trailing activation (trail_act), buffer (trail_buf), and historical win probability |
| `stock_screener.py` | Standalone script (independent of app.py's process) — queries TradingView's scanner API directly for a multi-filter weekly screen, sent to Telegram. Meant to be run via cron, not folded into app.py's scheduler (see its module docstring) |
| `earnings_estimates.json` | Unused — was consensus-estimate input for the old PDF-parsing earnings pipeline; `check_earnings_surprises` now gets estimates directly from TradingView |
| `portfolio_state.json` | Active trades: entry price, entry time, highest price (for trailing stop trigger) — gitignored, runtime-generated |
| `closed_trades.json` | Completed trades: entry/exit prices, PnL, exit reason — gitignored, runtime-generated |
| `seen_news.json` / `seen_earnings.json` | Dedup caches preventing repeat news/earnings alerts — gitignored, runtime-generated |
| `signal_reminders.json` | Next-day reminders of yesterday's BUY/SELL signals — gitignored, runtime-generated |
| `market_crash_state.json` | Last date a market-crash alert fired, so it only fires once per day — gitignored, runtime-generated |
| `fii_dii_history.json` | Daily FII net-flow history, used to compute the buy/sell streak shown in the pre-market brief — gitignored, runtime-generated |
| `.env` | Secrets: Telegram bot token, chat ID, optional Twilio credentials, optional HuggingFace token |
| `wfo/` | Separate, gitignored walk-forward-optimization research toolkit (not part of the deployed bot; has its own `.venv`) |

### Strategy Logic (in `app.py`)

**Entry Conditions (both must be true):**
1. MACD histogram "cooled off" — value falls within defined min/max range
2. Price above SMA(50) × (1 + min_pct_above_sma threshold)
3. No existing position (one trade at a time)

**Exit Conditions:**
- **Take Profit**: triggered at entry_price × (1 + tp%)
- **Stop Loss**: entry_price × (1 - sl%)
- **Trailing Stop** (optional): 
  - Sits dormant at -SL% until trade's highest price reaches entry_price × (1 + trail_activation%)
  - Then moves ONCE to entry_price × (1 + trail_buffer%) and locks there (does not continuously trail)
  - Only one position open at a time

**News Sentiment**:
- FinBERT (`ProsusAI/finbert`) via the free HuggingFace Inference API when `HF_API_TOKEN` is set; falls back to keyword matching (`POSITIVE_WORDS`/`NEGATIVE_WORDS`) if the token is missing or the API errors
- Fetches per-stock news from `yfinance`, falling back to Google News RSS
- Deduplicates using title similarity threshold (0.6 Jaccard index)
- Attaches sentiment to buy/sell alerts and to the standalone breaking-news stream

**Pre-market brief data sourcing** (all directly callable from a standalone script — no MCP/interactive tools involved, since this runs unattended on a schedule):
- Global cues / commodities / USD-INR / India VIX / Nifty close → `yfinance` tickers (`^N225`, `^HSI`, `^DJI`, `^IXIC`, `CL=F`, `GC=F`, `INR=X`, `^INDIAVIX`, `^NSEI`)
- FII/DII flows and Nifty-50 breadth → NSE's own JSON endpoints (`/api/fiidiiTradeReact`, `/api/allIndices`), via the cookie-priming session helpers `get_nse_session`/`_prime_nse_cookies` (in `app.py`'s earnings section, historically — earnings itself no longer uses NSE, see below) — NSE's APIs 401/403 without cookies from a prior page load
- GIFT Nifty → no free structured API exists for it anywhere (checked `yfinance`, NSE's public endpoints, and third-party MCP tools); it's regex text-mined from fresh (≤12h) Google News headlines instead, and the line is simply omitted if nothing parseable is found — never fabricated
- Nifty-50 breadth is intentionally labelled as such, not "market breadth" — NSE has no public full-market breadth endpoint

**Earnings-surprise and screener data sourcing** — both use TradingView's undocumented scanner API (`scanner.tradingview.com/india/scan`, no auth required) rather than NSE:
- `check_earnings_surprises` batches all tracked tickers into one scanner query (`name` filter with `in_range`) instead of polling NSE per-stock — pulls `earnings_per_share_diluted_fq`/`_forecast_fq` and `total_revenue_fq`/`revenue_forecast_fq`, i.e. TradingView's own actual/estimate figures, not values computed locally
- `stock_screener.py` filters server-side via the scanner's `filter` array (including direct column-vs-column comparisons, e.g. `close > EMA50`) — one request screens the whole India market, no per-stock looping
- Field names for both were confirmed against the scanner's own `/india/metainfo` field list and cross-checked against live values before use, not guessed

## Development Commands

### Run the Bot
```bash
python app.py
```
- Performs immediate analysis of all configured stocks plus one run of every other scheduled check
- Then blocks in the scheduler loop (hourly trading checks, 30-min news, 15-min earnings, 5-min crash checks, daily 9:08 AM pre-market brief, Friday 4 PM weekly report — see `run_scheduler()`)
- Sends Telegram (and, if configured, WhatsApp) alerts
- Logs to `bot.log`

### Backtest & Optimize Parameters
```bash
python strategy_optimizer.py
```
- Operates on **one ticker at a time**, configured by editing the constants near the top of the file (`TICKER`, `START_DATE`, `DATA_SOURCE` — `"nse"`/`"tv"`/`"yfinance"`/`"csv"`) — there are no CLI args
- Grid-searches exit parameters against that ticker's full history; target thresholds: Win Rate ≥ 80%, Profit Factor ≥ 2.0 (`TARGET_WIN_RATE`, `TARGET_PROFIT_FACTOR`)
- Discards combos below `MIN_TRADES` to avoid mistaking small-sample noise for edge
- Runs an out-of-sample walk-forward check on the best in-sample combo (`TRAIN_FRACTION` split) — if out-of-sample results fall apart, treat the combo as overfit
- Outputs the full grid to `optimization_results.csv`
- **Use this before adding a new stock or changing exit parameters** in `stocks_config.json`

### Run the Weekly Stock Screener
```bash
python stock_screener.py
```
- Standalone, independent of `app.py`'s process — runs once and exits, sends matches to Telegram
- Filters are hardcoded constants near the top of the file (`FILTERS` list + the `MIN_*`/`MAX_*` thresholds above it) — no CLI args
- Intended to be scheduled via cron, not run continuously; see the module docstring for the exact cron line

### Test TradingView Data Feed
```bash
python test_tv.py
```
- Quick connectivity check for `tvdatafeed` API

## Configuration

### Environment Variables (`.env`)
```
TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>
TWILIO_ACCOUNT_SID=<optional>
TWILIO_AUTH_TOKEN=<optional>
TWILIO_API_KEY=<optional>
TWILIO_API_SECRET=<optional>
TWILIO_FROM_WHATSAPP=<optional>
TWILIO_TO_WHATSAPP=<optional>
HF_API_TOKEN=<optional — free token from huggingface.co/settings/tokens, enables FinBERT sentiment instead of the keyword fallback>
```
`clean_env_var()` strips inline `#` comments from any of these, so values can be commented in place in `.env`.

### Stock Configuration (`stocks_config.json`)
Each stock entry requires:
- `exchange`: "NSE" (or other; currently NSE only)
- `name`: display name for alerts
- `tp`: take-profit percentage (e.g., 0.25 = 25%)
- `sl`: stop-loss percentage (e.g., 0.2 = 20%)
- `trail_act`: trailing stop activation level as % gain (e.g., 0.17 = 17% above entry)
- `trail_buf`: trailing stop buffer once activated (e.g., 0.08 = 8% below the high-water mark)
- `probability`: historical backtest win rate (informational; does not affect trading)
- `yf_ticker`: optional override when the yfinance symbol differs from the NSE ticker key (e.g. `BAJAJ-AUTO.NS` for key `BAJAJ_AUTO`, or `^NSEI` for the `NIFTY` index entry)

**To add a new stock:**
1. Backtest it first: edit `TICKER`/`START_DATE` in `strategy_optimizer.py` and run it
2. Add entry to `stocks_config.json` with optimized parameters
3. Restart the bot

## Important Notes

### Data Source Priority
- Primary: `tvdatafeed` (TradingView charts data) — used for the core MACD/SMA daily-bar analysis
- Secondary: `yfinance` — used for per-stock news, breaking-news stream, and all pre-market-brief data (global cues, commodities, VIX, FX, Nifty level)
- TradingView's scanner API (`scanner.tradingview.com`, separate from `tvdatafeed`) — used for earnings-surprise data and the weekly screener; unauthenticated but undocumented
- NSE's own JSON APIs (`nseindia.com/api/...`) — used for FII/DII flows and Nifty-50 breadth in the pre-market brief; these require a primed cookie session (see `get_nse_session`) or they 401/403

### State Persistence
- `portfolio_state.json`: tracks active trades; essential for resuming after restarts
- Safe to manually inspect or reset (the bot will rebuild it on next run)
- All the other runtime JSON files (dedup caches, alert-history state) follow the same "safe to delete, bot rebuilds it" pattern — deleting one just means the corresponding alert may briefly re-fire or a streak/dedup counter resets to zero

### Known Constraints
- **One position per stock, long-only** — no shorting, no pyramiding
- **Market hours only** — strategy assumes NSE trading hours (9:15 AM – 3:30 PM IST); most scheduled checks call `is_market_closed()` (weekends + the hardcoded `NSE_HOLIDAYS` calendar, currently populated for 2026) and no-op on closed days
- **Trailing stop behavior**: Does not continuously trail; it "steps" once when activation is reached, then locks
- **News deduplication**: 60% Jaccard similarity threshold; identical articles within 24 hours are skipped
- **`twilio` is imported unconditionally** (`from twilio.rest import Client`) but is **not** in `requirements.txt` — installing only from `requirements.txt` will fail at import time even if you don't use WhatsApp alerts; install it separately or add it to `requirements.txt`
- **NSE and TradingView endpoints are both unofficial** (undocumented, no auth) and can block/rate-limit; every dependent function (`fetch_fii_dii_flows`, `fetch_nifty50_breadth`, `fetch_earnings_data`, `stock_screener.py`'s `fetch_screener_matches`) degrades gracefully to "data unavailable" / skip-this-cycle rather than crashing. Field names for both were verified empirically against live responses (TradingView's `scanner.tradingview.com/india/metainfo` lists all valid scanner fields) rather than guessed — re-verify before trusting a new field name if one is ever added

### Common Modifications
- **Change monitoring frequency**: edit the relevant `schedule.every(...)` line in `run_scheduler()` (e.g. `schedule.every(1).hours.do(analyze_stocks)` → `.minutes` for faster testing)
- **Adjust MACD parameters**: `MACD_FAST`, `MACD_SLOW`, `MACD_SIG`, `HIST_MIN`, `HIST_MAX`, `SMA_LEN`, `SMA_PCT` constants near the top of `app.py` — keep `MACD+SMA.pinescript` and `strategy_optimizer.py` in sync if you change these
- **Tweak sentiment weights**: `POSITIVE_WORDS` and `NEGATIVE_WORDS` sets (keyword-fallback path only; FinBERT is model-based and unaffected)
- **Adjust the market-crash threshold**: `MARKET_CRASH_PCT` constant (currently -2.0)
- **Adjust the earnings-surprise threshold**: `EARNINGS_SURPRISE_THRESHOLD` constant (currently 5%)

## Dependencies

See `requirements.txt`:
- `pandas`, `requests`, `feedparser` — data processing & APIs
- `python-dotenv` — environment variable loading
- `schedule` — task scheduling
- `tvdatafeed` — TradingView market data (from GitHub fork)
- `yfinance` — Yahoo Finance fallback

Not in `requirements.txt` but required at import time:
- `twilio` — see the constraint noted above

Install with: `pip install -r requirements.txt` (then `pip install twilio` separately, or add it to the file).
