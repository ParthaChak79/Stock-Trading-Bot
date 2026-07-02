# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Telegram Trading Bot — AI Wealth Builder Strategy**

A Python application that monitors NSE (National Stock Exchange) stocks and sends automated BUY/SELL trading alerts via Telegram using the **AI Wealth Builder Strategy**. The bot uses MACD (Moving Average Convergence Divergence) with cooldown mechanics and SMA 50 trend filtering for entry signals, paired with customizable trailing stop configurations per stock for exits.

Key features:
- Real-time stock monitoring with hourly analysis
- MACD-based entry signals with cooldown and trend filtering
- Per-stock trailing stop and take-profit configuration
- News sentiment analysis and delivery via Telegram
- Portfolio state persistence across restarts (saved in `portfolio_state.json`)
- Closed trade history tracking (saved in `closed_trades.json`)

## Architecture & Key Components

### Data Flow
1. **Market Data Ingestion** → `tvdatafeed` (TradingView data) or `yfinance` as fallback
2. **Signal Generation** → MACD/SMA calculations in `app.py`
3. **Position Management** → Entry/exit logic with trailing stops
4. **Notifications** → Telegram alerts + Google News RSS sentiment analysis
5. **State Persistence** → JSON files track open/closed trades across restarts

### Core Files

| File | Purpose |
|------|---------|
| `app.py` | Main bot engine; contains all strategy logic, state management, and notification dispatch |
| `strategy_optimizer.py` | Backtest harness for grid-searching optimal exit parameters (TP%, SL%, trail activation%, trail buffer%) |
| `stocks_config.json` | Per-stock configuration: take-profit (tp), stop-loss (sl), trailing activation (trail_act), buffer (trail_buf), and historical win probability |
| `portfolio_state.json` | Active trades: entry price, entry time, highest price (for trailing stop trigger), stop/target prices |
| `closed_trades.json` | Completed trades: entry/exit prices, PnL, exit reason |
| `seen_news.json` | Deduplication cache: prevents re-sending the same news article |
| `.env` | Secrets: Telegram bot token, chat ID, optional Twilio credentials for WhatsApp alerts |

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

**News Sentiment** (optional enhancement):
- Fetches Google News RSS for each stock
- Classifies headlines as positive/negative/neutral based on keyword matching
- Deduplicates using title similarity threshold (0.6 Jaccard index)
- Attaches sentiment to buy/sell alerts

## Development Commands

### Run the Bot
```bash
python app.py
```
- Performs immediate analysis of all configured stocks
- Schedules hourly recurring checks (while script is active)
- Sends Telegram alerts on BUY/SELL signals
- Logs to `bot.log`

### Backtest & Optimize Parameters
```bash
python strategy_optimizer.py
```
- Grid-searches exit parameters for all stocks in `stocks_config.json`
- Target thresholds: Win Rate ≥ 80%, Profit Factor ≥ 2.0
- Outputs results to `optimization_results.csv`
- **Use this before updating `stocks_config.json` parameters** to validate a new stock or parameter set

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
```

### Stock Configuration (`stocks_config.json`)
Each stock entry requires:
- `exchange`: "NSE" (or other; currently NSE only)
- `name`: display name for alerts
- `tp`: take-profit percentage (e.g., 0.25 = 25%)
- `sl`: stop-loss percentage (e.g., 0.2 = 20%)
- `trail_act`: trailing stop activation level as % gain (e.g., 0.17 = 17% above entry)
- `trail_buf`: trailing stop buffer once activated (e.g., 0.08 = 8% below the high-water mark)
- `probability`: historical backtest win rate (informational; does not affect trading)

**To add a new stock:**
1. Backtest it first: run `python strategy_optimizer.py` with the new ticker
2. Add entry to `stocks_config.json` with optimized parameters
3. Restart the bot

## Important Notes

### Data Source Priority
- Primary: `tvdatafeed` (TradingView charts data) — most reliable for intraday
- Fallback: `yfinance` (Yahoo Finance) — if TradingView is unavailable
- Both provide 1-minute OHLC bars for the strategy

### State Persistence
- `portfolio_state.json`: tracks active trades; essential for resuming after restarts
- Always backed up implicitly — losing it only affects position tracking for the current trade window
- Safe to manually inspect or reset (the bot will rebuild it on next run)

### Known Constraints
- **One position per stock, long-only** — no shorting, no pyramiding
- **Market hours only** — strategy assumes NSE trading hours (9:15 AM – 3:30 PM IST)
- **Trailing stop behavior**: Does not continuously trail; it "steps" once when activation is reached, then locks
- **News deduplication**: 60% Jaccard similarity threshold; identical articles within 24 hours are skipped

### Common Modifications
- **Change monitoring frequency**: Edit `schedule.every(1).hours.do(analyze_and_trade)` in `app.py` (e.g., `.minutes` for faster testing)
- **Adjust MACD parameters**: See `MACD_FAST`, `MACD_SLOW`, `MACD_SIG` constants in `app.py` (line ~98)
- **Tweak sentiment weights**: `POSITIVE_WORDS` and `NEGATIVE_WORDS` sets determine what news keyword matches are positive/negative
- **Disable news alerts**: Comment out the news fetching section in the trade notification logic

## Dependencies

See `requirements.txt`:
- `pandas`, `requests`, `feedparser` — data processing & APIs
- `python-dotenv` — environment variable loading
- `schedule` — task scheduling
- `tvdatafeed` — TradingView market data (from GitHub fork)
- `yfinance` — Yahoo Finance fallback
- `twilio` (optional, listed implicitly) — WhatsApp notifications

Install with: `pip install -r requirements.txt`
