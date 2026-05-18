import pandas as pd
import json
import os
import requests
import feedparser
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv
from tvDatafeed import TvDatafeed, Interval

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")
SEEN_NEWS_FILE = os.path.join(BASE_DIR, "seen_news.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Load env variables
load_dotenv(ENV_FILE)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Strategy Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
HIST_MIN = -10.0
HIST_MAX = 2.0
SMA_LEN = 50
SMA_PCT = 0.02 # Minimum 2% above 50 SMA

# Stocks Configuration (TradingView Symbols & Exchanges)
STOCKS = {
    "BRITANNIA": {"exchange": "NSE", "name": "Britannia", "tp": 0.25, "sl": 0.20, "trail_act": 0.16, "trail_buf": 0.08},
    "EPL": {"exchange": "NSE", "name": "EPL", "tp": 0.29, "sl": 0.21, "trail_act": 0.06, "trail_buf": 0.04},
    "APOLLOHOSP": {"exchange": "NSE", "name": "Apollo Hospitals", "tp": 0.27, "sl": 0.21, "trail_act": 0.06, "trail_buf": 0.04},
    "BHARTIARTL": {"exchange": "NSE", "name": "Bharti Airtel", "tp": 0.30, "sl": 0.21, "trail_act": 0.06, "trail_buf": 0.04},
    "TORNTPOWER": {"exchange": "NSE", "name": "Torrent Power", "tp": 0.29, "sl": 0.18, "trail_act": 0.06, "trail_buf": 0.03},
    "PIDILITIND": {"exchange": "NSE", "name": "Pidilite", "tp": 0.28, "sl": 0.17, "trail_act": 0.08, "trail_buf": 0.05},
    "NATCOPHARM": {"exchange": "NSE", "name": "Natco Pharma", "tp": 0.26, "sl": 0.14, "trail_act": 0.08, "trail_buf": 0.04},
    "TVSMOTOR": {"exchange": "NSE", "name": "TVS Motors", "tp": 0.28, "sl": 0.14, "trail_act": 0.10, "trail_buf": 0.06},
    "BEL": {"exchange": "NSE", "name": "Bharat Electronics", "tp": 0.33, "sl": 0.14, "trail_act": 0.33, "trail_buf": 0.00},
    "GODREJCP": {"exchange": "NSE", "name": "Godrej Consumer Products", "tp": 0.25, "sl": 0.13, "trail_act": 0.25, "trail_buf": 0.00},
    "SCHNEIDER": {"exchange": "NSE", "name": "Schneider Electric Infrastructure", "tp": 0.25, "sl": 0.22, "trail_act": 0.16, "trail_buf": 0.04},
    "FORTIS": {"exchange": "NSE", "name": "Fortis Healthcare", "tp": 0.25, "sl": 0.24, "trail_act": 0.13, "trail_buf": 0.07},
    "MAXHEALTH": {"exchange": "NSE", "name": "Max Healthcare", "tp": 0.25, "sl": 0.21, "trail_act": 0.10, "trail_buf": 0.04},
    "LT": {"exchange": "NSE", "name": "Larsen & Toubro", "tp": 0.24, "sl": 0.19, "trail_act": 0.10, "trail_buf": 0.04},
    "HAL": {"exchange": "NSE", "name": "Hindustan Aeronautics", "tp": 0.27, "sl": 0.17, "trail_act": 0.08, "trail_buf": 0.05},
}

# Initialize TradingView Datafeed
print("Initializing TradingView Connection...")
tv = TvDatafeed()

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def load_seen_news():
    if os.path.exists(SEEN_NEWS_FILE):
        with open(SEEN_NEWS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen_news(seen_news):
    with open(SEEN_NEWS_FILE, "w") as f:
        json.dump(list(seen_news), f)

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- Telegram credentials missing! Please configure .env file ---")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending telegram message: {e}")

def get_news(stock_name):
    """Fetch top 2 recent news articles for the stock (for trade alerts)"""
    query = stock_name.replace(' ', '+') + "+stock+news+india"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries[:2]:
            news_items.append(f"• <a href='{entry.link}'>{entry.title}</a>")
        if news_items:
            return "\n".join(news_items)
        return "No recent news found."
    except Exception as e:
        return f"Could not fetch news: {e}"

def check_news_stream():
    """Continuously checks for breaking news across all sources and alerts via Telegram"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for breaking news...")
    seen_news = load_seen_news()
    new_articles = False
    
    for ticker, config in STOCKS.items():
        query = config['name'].replace(' ', '+') + "+stock+news+india"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            feed = feedparser.parse(url)
            # Check top 3 articles
            for entry in feed.entries[:3]:
                if entry.link not in seen_news:
                    seen_news.add(entry.link)
                    new_articles = True
                    
                    msg = f"📰 <b>Breaking News: {config['name']}</b>\n\n"
                    msg += f"<b>{entry.title}</b>\n\n"
                    msg += f"<a href='{entry.link}'>🔗 Read Full Article</a>\n"
                    
                    send_telegram_message(msg)
                    print(f"Sent News Alert for {ticker}")
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            
    if new_articles:
        save_seen_news(seen_news)

def calculate_indicators(df):
    """Calculate MACD and SMA 50"""
    if len(df) < SMA_LEN:
        return None # Not enough data
    
    df['SMA_50'] = df['close'].rolling(window=SMA_LEN).mean()
    
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIG, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

def analyze_stocks():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running TradingView market analysis...")
    state = load_state()
    
    for ticker, config in STOCKS.items():
        try:
            # Fetch 200 bars of daily data from TradingView
            df = tv.get_hist(symbol=ticker, exchange=config['exchange'], interval=Interval.in_daily, n_bars=200)
            
            if df is None or df.empty:
                print(f"No data fetched for {ticker}")
                continue
                
            df = calculate_indicators(df)
            if df is None:
                print(f"Not enough data for {ticker} to calculate indicators.")
                continue
                
            latest = df.iloc[-1]
            current_close = latest['close']
            current_high = latest['high']
            hist_line = latest['MACD_Hist']
            sma_50 = latest['SMA_50']
            
            # Format date beautifully
            date_str = latest.name.strftime("%d %b %Y")
            
            if ticker in state:
                # --- Manage Active Trade ---
                trade = state[ticker]
                entry_price = trade['entry_price']
                highest_price = trade['highest_price']
                
                # Update Highest Price for Trailing Stop
                if current_high > highest_price:
                    highest_price = current_high
                    trade['highest_price'] = highest_price
                    save_state(state) 
                
                # Exit levels
                target_price = entry_price * (1 + config['tp'])
                activation_price = entry_price * (1 + config['trail_act'])
                
                # Trailing logic
                if highest_price >= activation_price:
                    stop_price = entry_price * (1 + config['trail_buf'])
                else:
                    stop_price = entry_price * (1 - config['sl'])
                    
                # Check for Exit Condition
                sell_reason = None
                if current_close >= target_price:
                    sell_reason = f"🎯 <b>TAKE PROFIT</b> Hit at ₹{current_close:.2f}"
                elif current_close <= stop_price:
                    if highest_price >= activation_price:
                        sell_reason = f"🛡️ <b>TRAILING STOP</b> Hit at ₹{current_close:.2f} (Locked in +{config['trail_buf']*100}% Buffer)"
                    else:
                        sell_reason = f"🛑 <b>STOP LOSS</b> Hit at ₹{current_close:.2f} (-{config['sl']*100}%)"
                        
                if sell_reason:
                    profit_pct = ((current_close - entry_price) / entry_price) * 100
                    
                    msg = f"📉 <b>SELL ALERT: {config['name']}</b>\n"
                    msg += f"🗓️ Date: {date_str}\n"
                    msg += f"💡 Reason: {sell_reason}\n\n"
                    msg += f"🚪 Entry Price: ₹{entry_price:.2f}\n"
                    msg += f"💵 Exit Price: ₹{current_close:.2f}\n"
                    msg += f"📊 Profit/Loss: <b>{profit_pct:.2f}%</b>\n\n"
                    msg += f"📰 <b>Recent News:</b>\n{get_news(config['name'])}"
                    
                    print(f"Sending SELL alert for {ticker} (Reason: {sell_reason})")
                    send_telegram_message(msg)
                    
                    # Remove from active trades
                    del state[ticker]
                    save_state(state)
                else:
                    # Logging ongoing trade
                    print(f"{ticker} [HOLD] - Current: ₹{current_close:.2f} | Entry: ₹{entry_price:.2f} | High: ₹{highest_price:.2f} | Stop: ₹{stop_price:.2f}")
                    
            else:
                # --- Check for New Buy Signal ---
                # 1. MACD Cooldown Condition 
                is_cooled_off = (hist_line > HIST_MIN) and (hist_line <= HIST_MAX)
                
                # 2. Trend Condition (Price minimum 2% above 50 SMA)
                is_trend_intact = current_close > (sma_50 * (1 + SMA_PCT))
                
                if is_cooled_off and is_trend_intact:
                    # Enter Trade
                    state[ticker] = {
                        "entry_price": current_close,
                        "highest_price": current_high,
                        "date": date_str
                    }
                    save_state(state)
                    
                    msg = f"🚀 <b>BUY ALERT: {config['name']}</b>\n"
                    msg += f"🗓️ Date: {date_str}\n\n"
                    msg += f"🟢 Entry Price: ₹{current_close:.2f}\n"
                    msg += f"🎯 Target Price: ₹{current_close * (1 + config['tp']):.2f} (+{config['tp']*100}%)\n"
                    msg += f"🛡️ Stop Loss: ₹{current_close * (1 - config['sl']):.2f} (-{config['sl']*100}%)\n"
                    msg += f"📈 Trail Activation: +{config['trail_act']*100}%\n"
                    msg += f"⚙️ Strategy: AI Wealth Builder (Trend + Cooldown)\n\n"
                    msg += f"📰 <b>Recent News:</b>\n{get_news(config['name'])}"
                    
                    print(f"Sending BUY alert for {ticker}")
                    send_telegram_message(msg)
                else:
                    print(f"{ticker} [WAIT] - Cooldown: {is_cooled_off} ({hist_line:.2f}), Trend Intact: {is_trend_intact}")
                    
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

def run_scheduler():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Telegram Trading Bot Started!")
    
    # Run once immediately on startup
    analyze_stocks()
    check_news_stream()
    
    # Schedule trading check every hour
    schedule.every(1).hours.do(analyze_stocks)
    
    # Schedule breaking news check every 30 minutes
    schedule.every(30).minutes.do(check_news_stream)
    
    print("\nScheduler running. Press Ctrl+C to exit.\n")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nExiting bot gracefully...")
            break

if __name__ == "__main__":
    run_scheduler()
