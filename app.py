import pandas as pd
import json
import os
import requests
import feedparser
import schedule
import time
import re
import string
import yfinance as yf
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tvDatafeed import TvDatafeed, Interval
from twilio.rest import Client

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")
SEEN_NEWS_FILE = os.path.join(BASE_DIR, "seen_news.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")

def clean_env_var(value):
    if not value:
        return value
    # Split by '#' to remove inline comments
    value = value.split('#')[0]
    return value.strip()

# Load env variables
load_dotenv(ENV_FILE)
TELEGRAM_BOT_TOKEN = clean_env_var(os.getenv("TELEGRAM_BOT_TOKEN"))
TELEGRAM_CHAT_ID = clean_env_var(os.getenv("TELEGRAM_CHAT_ID"))
TWILIO_ACCOUNT_SID = clean_env_var(os.getenv("TWILIO_ACCOUNT_SID"))
TWILIO_AUTH_TOKEN = clean_env_var(os.getenv("TWILIO_AUTH_TOKEN"))
TWILIO_API_KEY = clean_env_var(os.getenv("TWILIO_API_KEY"))
TWILIO_API_SECRET = clean_env_var(os.getenv("TWILIO_API_SECRET"))
TWILIO_FROM_WHATSAPP = clean_env_var(os.getenv("TWILIO_FROM_WHATSAPP"))
TWILIO_TO_WHATSAPP = clean_env_var(os.getenv("TWILIO_TO_WHATSAPP"))

# Global Timezones & Startup time
IST = timezone(timedelta(hours=5, minutes=30))
STARTUP_TIME = datetime.now(timezone.utc)

# Sentiment vocabulary
POSITIVE_WORDS = {
    'profit', 'rise', 'rises', 'growth', 'grow', 'jump', 'jumps', 'surge', 'surges', 'gain', 'gains', 'upbeat', 'positive',
    'bullish', 'expansion', 'expand', 'highest', 'record', 'exceed', 'exceeds', 'beat', 'beats', 'soar', 'soars', 'win', 'wins',
    'buy', 'strong', 'outperform', 'upgrade', 'upgraded', 'recovery', 'recover', 'demand', 'revenue increase', 'acquisition',
    'dividend', 'breakout', 'momentum', 'approval', 'contract', 'partnership', 'bonus', 'all-time high', 'bull'
}
NEGATIVE_WORDS = {
    'loss', 'fall', 'falls', 'drop', 'drops', 'decline', 'declines', 'dip', 'dips', 'plunge', 'plunges', 'slump', 'slumps',
    'negative', 'bearish', 'weak', 'weakness', 'down', 'cut', 'cuts', 'shrink', 'shrinks', 'hit', 'hits', 'deficit',
    'fail', 'fails', 'miss', 'misses', 'debt', 'risk', 'risks', 'warn', 'warns', 'warning', 'sell', 'underperform',
    'downgrade', 'downgraded', 'disruption', 'disruptions', 'pressure', 'pressures', 'slid', 'slide', 'slides',
    'lawsuit', 'resignation', 'fraud', 'probe', 'inflation', 'recession', 'bear', 'default', 'penalty'
}

def clean_title(title):
    title = re.sub(r'\s+[-|]\s+.*$', '', title)
    title = title.lower()
    title = title.translate(str.maketrans('', '', string.punctuation))
    return title.strip()

def is_duplicate_title(new_title, seen_titles, threshold=0.6):
    new_cleaned = clean_title(new_title)
    new_words = set(new_cleaned.split())
    if not new_words:
        return False
    for seen in seen_titles:
        seen_cleaned = clean_title(seen)
        seen_words = set(seen_cleaned.split())
        if not seen_words:
            continue
        intersection = new_words.intersection(seen_words)
        union = new_words.union(seen_words)
        similarity = len(intersection) / len(union)
        if similarity >= threshold:
            return True
    return False

def analyze_sentiment(title, summary):
    title_text = title.lower()
    summary_text = (summary or '').lower()
    
    # Weight title 2x
    pos_count = sum(2 if word in title_text else 1 for word in POSITIVE_WORDS if word in title_text or word in summary_text)
    neg_count = sum(2 if word in title_text else 1 for word in NEGATIVE_WORDS if word in title_text or word in summary_text)
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

# Strategy Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
HIST_MIN = -10.0
HIST_MAX = 2.0
SMA_LEN = 50
SMA_PCT = 0.02 # Minimum 2% above 50 SMA

# NSE Market Holidays Configuration for 2026
NSE_HOLIDAYS = {
    2026: {
        "01-15", # Municipal Corporation Election - Maharashtra
        "01-26", # Republic Day
        "03-03", # Holi
        "03-26", # Shri Ram Navami
        "03-31", # Shri Mahavir Jayanti
        "04-03", # Good Friday
        "04-14", # Dr. Baba Saheb Ambedkar Jayanti
        "05-01", # Maharashtra Day
        "05-28", # Bakri Id
        "06-26", # Muharram
        "09-14", # Ganesh Chaturthi
        "10-02", # Mahatma Gandhi Jayanti
        "10-20", # Dussehra
        "11-10", # Diwali-Balipratipada
        "11-24", # Prakash Gurpurb Sri Guru Nanak Dev
        "12-25"  # Christmas
    }
}

def is_market_closed(date_obj):
    # Saturday = 5, Sunday = 6
    if date_obj.weekday() >= 5:
        return True
    year = date_obj.year
    date_str = date_obj.strftime("%m-%d")
    if year in NSE_HOLIDAYS:
        if date_str in NSE_HOLIDAYS[year]:
            return True
    return False

# Stocks Configuration (Loaded Dynamically from stocks_config.json)
CONFIG_FILE = os.path.join(BASE_DIR, "stocks_config.json")

def load_stocks_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading stocks_config.json: {e}")
    # Fallback to hardcoded dict if file doesn't exist
    return {
        "BRITANNIA": {"exchange": "NSE", "name": "Britannia", "tp": 0.25, "sl": 0.20, "trail_act": 0.17, "trail_buf": 0.08},
        "EPL": {"exchange": "NSE", "name": "EPL", "tp": 0.27, "sl": 0.26, "trail_act": 0.18, "trail_buf": 0.09},
        "APOLLOHOSP": {"exchange": "NSE", "name": "Apollo Hospitals", "tp": 0.27, "sl": 0.23, "trail_act": 0.15, "trail_buf": 0.09},
        "BHARTIARTL": {"exchange": "NSE", "name": "Bharti Airtel", "tp": 0.30, "sl": 0.23, "trail_act": 0.14, "trail_buf": 0.07},
        "TORNTPOWER": {"exchange": "NSE", "name": "Torrent Power", "tp": 0.29, "sl": 0.23, "trail_act": 0.12, "trail_buf": 0.08},
        "PIDILITIND": {"exchange": "NSE", "name": "Pidilite", "tp": 0.28, "sl": 0.23, "trail_act": 0.13, "trail_buf": 0.07},
        "NATCOPHARM": {"exchange": "NSE", "name": "Natco Pharma", "tp": 0.27, "sl": 0.23, "trail_act": 0.13, "trail_buf": 0.09},
        "TVSMOTOR": {"exchange": "NSE", "name": "TVS Motors", "tp": 0.28, "sl": 0.24, "trail_act": 0.14, "trail_buf": 0.08},
        "BEL": {"exchange": "NSE", "name": "Bharat Electronics", "tp": 0.33, "sl": 0.24, "trail_act": 0.17, "trail_buf": 0.07},
        "GODREJCP": {"exchange": "NSE", "name": "Godrej Consumer Products", "tp": 0.29, "sl": 0.24, "trail_act": 0.17, "trail_buf": 0.09},
        "SCHNEIDER": {"exchange": "NSE", "name": "Schneider Electric Infrastructure", "tp": 0.25, "sl": 0.22, "trail_act": 0.16, "trail_buf": 0.05},
        "FORTIS": {"exchange": "NSE", "name": "Fortis Healthcare", "tp": 0.25, "sl": 0.24, "trail_act": 0.13, "trail_buf": 0.07},
        "MAXHEALTH": {"exchange": "NSE", "name": "Max Healthcare", "tp": 0.25, "sl": 0.21, "trail_act": 0.10, "trail_buf": 0.04},
        "LT": {"exchange": "NSE", "name": "Larsen & Toubro", "tp": 0.24, "sl": 0.19, "trail_act": 0.10, "trail_buf": 0.04},
        "HAL": {"exchange": "NSE", "name": "Hindustan Aeronautics", "tp": 0.27, "sl": 0.17, "trail_act": 0.08, "trail_buf": 0.05},
        "HDFCBANK": {"exchange": "NSE", "name": "HDFC Bank", "tp": 0.27, "sl": 0.23, "trail_act": 0.15, "trail_buf": 0.08},
        "ICICIBANK": {"exchange": "NSE", "name": "ICICI Bank", "tp": 0.30, "sl": 0.25, "trail_act": 0.14, "trail_buf": 0.08},
        "DIXON": {"exchange": "NSE", "name": "Dixon Tech", "tp": 0.27, "sl": 0.19, "trail_act": 0.14, "trail_buf": 0.08},
        "BAJAJ_AUTO": {"exchange": "NSE", "name": "Bajaj Auto", "tp": 0.26, "sl": 0.19, "trail_act": 0.14, "trail_buf": 0.07, "yf_ticker": "BAJAJ-AUTO.NS"},
        "M&M": {"exchange": "NSE", "name": "M&M", "tp": 0.24, "sl": 0.15, "trail_act": 0.15, "trail_buf": 0.10},
        "ONGC": {"exchange": "NSE", "name": "ONGC", "tp": 0.25, "sl": 0.23, "trail_act": 0.09, "trail_buf": 0.05},
        "SBIN": {"exchange": "NSE", "name": "SBI", "tp": 0.26, "sl": 0.23, "trail_act": 0.14, "trail_buf": 0.06},
        "DIVISLAB": {"exchange": "NSE", "name": "Divi's Lab", "tp": 0.29, "sl": 0.16, "trail_act": 0.15, "trail_buf": 0.11},
        "POLYCAB": {"exchange": "NSE", "name": "Polycab", "tp": 0.30, "sl": 0.19, "trail_act": 0.17, "trail_buf": 0.09},
        "POWERGRID": {"exchange": "NSE", "name": "Power Grid", "tp": 0.25, "sl": 0.17, "trail_act": 0.14, "trail_buf": 0.08},
        "WABAG": {"exchange": "NSE", "name": "VA Tech Wabag", "tp": 0.26, "sl": 0.16, "trail_act": 0.12, "trail_buf": 0.08},
        "CDSL": {"exchange": "NSE", "name": "CDSL", "tp": 0.22, "sl": 0.18, "trail_act": 0.15, "trail_buf": 0.09},
        "KAYNES": {"exchange": "NSE", "name": "Kaynes Technology", "tp": 0.23, "sl": 0.13, "trail_act": 0.15, "trail_buf": 0.10},
        "PIIND": {"exchange": "NSE", "name": "PI Industries", "tp": 0.20, "sl": 0.17, "trail_act": 0.16, "trail_buf": 0.08},
        "ASTRAMICRO": {"exchange": "NSE", "name": "Astra Microwave Products", "tp": 0.29, "sl": 0.25, "trail_act": 0.12, "trail_buf": 0.08},
        "NIFTY": {"exchange": "NSE", "name": "NIFTY50 Index", "tp": 0.25, "sl": 0.24, "trail_act": 0.13, "trail_buf": 0.07, "yf_ticker": "^NSEI"},
        "KEI": {"exchange": "NSE", "name": "KEI Industries Limited", "tp": 0.26, "sl": 0.20, "trail_act": 0.15, "trail_buf": 0.05, "yf_ticker": "KEI.NS"},
        "NAVINFLUOR": {"exchange": "NSE", "name": "Navin Fluorine International Limited", "tp": 0.19, "sl": 0.185, "trail_act": 0.10, "trail_buf": 0.07, "yf_ticker": "NAVINFLUOR.NS"},
        "ZYDUSLIFE": {"exchange": "NSE", "name": "Zydus Lifesciences Limited", "tp": 0.18, "sl": 0.19, "trail_act": 0.08, "trail_buf": 0.06, "yf_ticker": "ZYDUSLIFE.NS"},
        "AJANTPHARM": {"exchange": "NSE", "name": "Ajanta Pharma", "tp": 0.21, "sl": 0.24, "trail_act": 0.09, "trail_buf": 0.10, "yf_ticker": "AJANTPHARM.NS"},
        "LUPIN": {"exchange": "NSE", "name": "Lupin Ltd", "tp": 0.15, "sl": 0.30, "trail_act": 0.12, "trail_buf": 0.11, "yf_ticker": "LUPIN.NS"},
        "RRKABEL": {"exchange": "NSE", "name": "RR Kabel Ltd", "tp": 0.10, "sl": 0.08, "trail_act": 0.08, "trail_buf": 0.06, "yf_ticker": "RRKABEL.NS"},
        "PRICOLLTD": {"exchange": "NSE", "name": "Pricol Ltd", "tp": 0.13, "sl": 0.17, "trail_act": 0.10, "trail_buf": 0.05, "yf_ticker": "PRICOLLTD.NS"},
        "THYROCARE": {"exchange": "NSE", "name": "Thyrocare", "tp": 0.15, "sl": 0.18, "trail_act": 0.08, "trail_buf": 0.03, "yf_ticker": "THYROCARE.NS"},
        "SJS": {"exchange": "NSE", "name": "SJS Enterprises", "tp": 0.25, "sl": 0.14, "trail_act": 0.13, "trail_buf": 0.11, "probability": 0.75, "yf_ticker": "SJS.NS"},
        "NH": {"exchange": "NSE", "name": "Narayana Hrudayalaya Ltd", "tp": 0.23, "sl": 0.26, "trail_act": 0.12, "trail_buf": 0.09, "probability": 0.92, "yf_ticker": "NH.NS"},
        "CAPLIPOINT": {"exchange": "NSE", "name": "Caplin Point Laboratories", "tp": 0.23, "sl": 0.26, "trail_act": 0.08, "trail_buf": 0.05, "probability": 0.81, "yf_ticker": "CAPLIPOINT.NS"},
        "MEDANTA": {"exchange": "NSE", "name": "Global Health Limited (Medanta)", "tp": 0.23, "sl": 0.18, "trail_act": 0.11, "trail_buf": 0.05, "probability": 0.85, "yf_ticker": "MEDANTA.NS"}
    }

STOCKS = load_stocks_config()

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
    seen_links = set()
    seen_titles = set()
    if os.path.exists(SEEN_NEWS_FILE):
        try:
            with open(SEEN_NEWS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    seen_links = set(data.get("links", []))
                    seen_titles = set(data.get("titles", []))
                elif isinstance(data, list):
                    seen_links = set(data)
        except Exception as e:
            print(f"Error loading seen_news: {e}")
    return seen_links, seen_titles

def save_seen_news(seen_links, seen_titles):
    try:
        with open(SEEN_NEWS_FILE, "w") as f:
            json.dump({
                "links": list(seen_links),
                "titles": list(seen_titles)
            }, f, indent=4)
    except Exception as e:
        print(f"Error saving seen_news: {e}")

CLOSED_TRADES_FILE = os.path.join(BASE_DIR, "closed_trades.json")

def load_closed_trades():
    if not os.path.exists(CLOSED_TRADES_FILE):
        # Seed with initial trades since May 2026 as requested
        initial_trades = [
            {
                "ticker": "NATCOPHARM",
                "name": "Natco Pharma",
                "entry_price": 1165.60,
                "exit_price": 1210.00,
                "entry_date": "15 May 2026",
                "exit_date": "22 May 2026",
                "pnl_pct": 3.81,
                "reason": "Trailing Stop Hit"
            },
            {
                "ticker": "WABAG",
                "name": "VA Tech Wabag",
                "entry_price": 1452.80,
                "exit_price": 1585.00,
                "entry_date": "25 May 2026",
                "exit_date": "18 Jun 2026",
                "pnl_pct": 9.10,
                "reason": "Take Profit Hit"
            }
        ]
        try:
            with open(CLOSED_TRADES_FILE, "w") as f:
                json.dump(initial_trades, f, indent=4)
            return initial_trades
        except Exception as e:
            print(f"Error seeding closed_trades.json: {e}")
            return []
            
    try:
        with open(CLOSED_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading closed_trades.json: {e}")
    return []

def log_closed_trade(ticker, name, entry_price, exit_price, entry_date, exit_date, pnl_pct, reason):
    trades = load_closed_trades()
    # Check duplicate
    for t in trades:
        if t['ticker'] == ticker and t['exit_date'] == exit_date:
            return
            
    trades.append({
        "ticker": ticker,
        "name": name,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "entry_date": entry_date,
        "exit_date": exit_date,
        "pnl_pct": float(pnl_pct),
        "reason": str(reason)
    })
    try:
        with open(CLOSED_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=4)
    except Exception as e:
        print(f"Error saving closed_trades.json: {e}")

def send_whatsapp_message(message):
    if not TWILIO_ACCOUNT_SID:
        print("\n[WhatsApp] Twilio Account SID missing! Skipping WhatsApp broadcast.")
        return
        
    if not TWILIO_FROM_WHATSAPP or not TWILIO_TO_WHATSAPP:
        print(f"\n[WhatsApp] Twilio WhatsApp numbers missing! FROM: {TWILIO_FROM_WHATSAPP}, TO: {TWILIO_TO_WHATSAPP}. Skipping WhatsApp broadcast.")
        return
        
    if "your_twilio" in TWILIO_FROM_WHATSAPP or "your_twilio" in TWILIO_TO_WHATSAPP:
        print("\n[WhatsApp] Twilio WhatsApp numbers contain default placeholders! Skipping WhatsApp broadcast.")
        return
        
    has_api_key = TWILIO_API_KEY and TWILIO_API_SECRET and "your_twilio" not in TWILIO_API_KEY and "your_twilio" not in TWILIO_API_SECRET
    
    if not (TWILIO_AUTH_TOKEN or has_api_key):
        print("\n[WhatsApp] Twilio credentials missing! Provide Auth Token or API Key/Secret. Skipping WhatsApp broadcast.")
        return
    
    # Strip HTML tags since WhatsApp doesn't support <a> or <b> tags directly.
    # WhatsApp uses *text* for bold and _text_ for italics.
    clean_message = message.replace("<b>", "*").replace("</b>", "*")
    clean_message = clean_message.replace("<i>", "_").replace("</i>", "_")
    clean_message = re.sub(r'<a href=\'(.*?)\'>(.*?)</a>', r'\2: \1', clean_message)
    clean_message = re.sub(r'<[^<]+?>', '', clean_message) # strip any remaining html

    try:
        if has_api_key:
            client = Client(TWILIO_API_KEY, TWILIO_API_SECRET, TWILIO_ACCOUNT_SID)
        else:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
        print(f"[WhatsApp] Sending message to {TWILIO_TO_WHATSAPP} from {TWILIO_FROM_WHATSAPP}...")
        client.messages.create(
            body=clean_message,
            from_=TWILIO_FROM_WHATSAPP,
            to=TWILIO_TO_WHATSAPP
        )
        print("[WhatsApp] WhatsApp message sent successfully.")
    except Exception as e:
        print(f"[WhatsApp] Error sending WhatsApp message: {e}")
        print("[WhatsApp] Note: If you are using Twilio Sandbox, remember that the session expires every 72 hours. You must send 'join <sandbox-keyword>' to your Twilio number from your phone to re-enable it.")

def send_telegram_message(message):
    # Send to WhatsApp as well (if configured)
    send_whatsapp_message(message)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- Telegram credentials missing! Please configure .env file ---")
        return False
    
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
        return True
    except Exception as e:
        print(f"Error sending telegram message: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

REMINDERS_FILE = os.path.join(BASE_DIR, "signal_reminders.json")

def load_reminders():
    if os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading reminders: {e}")
    return []

def save_reminders(reminders):
    try:
        with open(REMINDERS_FILE, "w") as f:
            json.dump(reminders, f, indent=4)
    except Exception as e:
        print(f"Error saving reminders: {e}")

def add_reminder(ticker, signal_type, message):
    reminders = load_reminders()
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    reminders.append({
        "ticker": ticker,
        "type": signal_type,
        "message": message,
        "date_triggered": today_str,
        "reminder_sent": False
    })
    save_reminders(reminders)

def check_and_send_reminders():
    print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Checking for yesterday's signal reminders...")
    reminders = load_reminders()
    today = datetime.now(IST).date()
    updated = False
    
    for r in reminders:
        if not r.get("reminder_sent", False):
            try:
                triggered_date = datetime.strptime(r["date_triggered"], "%Y-%m-%d").date()
                if today > triggered_date:
                    orig_msg = r.get("message", "")
                if orig_msg:
                    # Prefix with a reminder label
                    reminder_msg = f"🔔 <b>YESTERDAY'S SIGNAL REMINDER</b> 🔔\n\n{orig_msg}"
                    if send_telegram_message(reminder_msg):
                        r["reminder_sent"] = True
                        updated = True
                        print(f"Sent reminder for {r['ticker']} {r['type']} signal from {r['date_triggered']}")
                    else:
                        print(f"Failed to send reminder for {r['ticker']}. Will retry next cycle.")
            except Exception as e:
                print(f"Error parsing triggered date for {r['ticker']}: {e}")
                
    if updated:
        save_reminders(reminders)

def send_holdings_report():
    print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Generating weekly holdings report...")
    state = load_state()
    closed_trades = load_closed_trades()
    
    if not state and not closed_trades:
        msg = "📊 <b>Weekly Holdings Report</b>\n\nNo active or exited share holdings from May 2026 onwards."
        send_telegram_message(msg)
        return
        
    msg_lines = ["━━━━━━━━━━━━━━━━━━━━━━\n📊 <b>Weekly Portfolio Report</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"]
    
    # 1. Active Holdings
    msg_lines.append("🟩 <b>Active Holdings</b>")
    if not state:
        msg_lines.append("<i>No active holdings at the moment.</i>\n")
    else:
        total_active_pnl_sum = 0
        active_count = 0
        for ticker, data in state.items():
            config = STOCKS.get(ticker)
            if not config:
                continue
                
            try:
                # Fetch latest daily data to get current price
                df = tv.get_hist(symbol=ticker, exchange=config['exchange'], interval=Interval.in_daily, n_bars=1)
                if df is not None and not df.empty:
                    current_price = df.iloc[-1]['close']
                else:
                    current_price = data['entry_price'] # fallback
            except Exception as e:
                print(f"Error fetching price for report for {ticker}: {e}")
                current_price = data['entry_price']
                
            entry_price = data['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            total_active_pnl_sum += pnl_pct
            active_count += 1
            
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            pnl_sign = "+" if pnl_pct >= 0 else ""
            
            msg_lines.append(
                f"• <b>{config.get('name', ticker)} ({ticker})</b>\n"
                f"  📅 Entry Date: {data['date']}\n"
                f"  🚪 Entry Price: ₹{entry_price:.2f}\n"
                f"  💵 Current Price: ₹{current_price:.2f}\n"
                f"  📊 P&L: {pnl_emoji} <b>{pnl_sign}{pnl_pct:.2f}%</b>\n"
            )
            
        if active_count > 0:
            avg_pnl = total_active_pnl_sum / active_count
            avg_sign = "+" if avg_pnl >= 0 else ""
            avg_emoji = "🟢" if avg_pnl >= 0 else "🔴"
            msg_lines.append(f"📈 <b>Average Active P&L:</b> {avg_emoji} <b>{avg_sign}{avg_pnl:.2f}%</b>\n")
            
    # 2. Exited Holdings
    msg_lines.append("━━━━━━━━━━━━━━━━━━━━━━\n🟥 <b>Recent Exits (Last 3 Months)</b>\n━━━━━━━━━━━━━━━━━━━━━━")
    
    # Filter out active tickers from exited list just in case
    active_tickers = set(state.keys())
    today_date = datetime.now(IST).date()
    
    exited_to_show = []
    for t in closed_trades:
        if t['ticker'] not in active_tickers:
            try:
                exit_date_obj = datetime.strptime(t['exit_date'], "%d %b %Y").date()
                if (today_date - exit_date_obj).days <= 90:
                    exited_to_show.append(t)
            except Exception:
                # If date parsing fails, include it just in case
                exited_to_show.append(t)
    
    if not exited_to_show:
        msg_lines.append("\n<i>No exited holdings to show.</i>")
    else:
        for t in exited_to_show:
            pnl_pct = t.get('pnl_pct', 0)
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            pnl_sign = "+" if pnl_pct >= 0 else ""
            
            # Extract plain text from reason (e.g. stripping HTML)
            reason_clean = re.sub('<[^<]+?>', '', t.get('reason', ''))
            
            msg_lines.append(
                f"\n• <b>{t.get('name', t['ticker'])} ({t['ticker']})</b>\n"
                f"  📅 Held: {t.get('entry_date', 'N/A')} to {t.get('exit_date', 'N/A')}\n"
                f"  🚪 Entry: ₹{t.get('entry_price', 0):.2f} | Exit: ₹{t.get('exit_price', 0):.2f}\n"
                f"  📊 Realized P&L: {pnl_emoji} <b>{pnl_sign}{pnl_pct:.2f}%</b> ({reason_clean})"
            )
            
    send_telegram_message("\n".join(msg_lines))

def get_news(stock_name, ticker=None):
    """Fetch top 2 recent news articles for the stock (for trade alerts)"""
    if not ticker:
        for tk, cfg in STOCKS.items():
            if cfg['name'] == stock_name:
                ticker = tk
                break
    if ticker:
        try:
            cfg = STOCKS.get(ticker, {})
            yf_ticker = cfg.get("yf_ticker", f"{ticker}.NS")
            t = yf.Ticker(yf_ticker)
            news = t.news
            news_items = []
            for article in news[:2]:
                content = article.get('content', {})
                title = content.get('title')
                summary = content.get('summary') or content.get('description') or ""
                link = content.get('clickThroughUrl', {}).get('url') or content.get('canonicalUrl', {}).get('url') or ""
                
                sentiment = analyze_sentiment(title, summary)
                emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
                
                summary_clean = re.sub('<[^<]+?>', '', summary).strip()
                summary_short = summary_clean[:180] + "..." if len(summary_clean) > 180 else summary_clean
                
                item = f"{emoji} <b>{title}</b>\n<i>{summary_short}</i>\n<a href='{link}'>🔗 Read Link</a>"
                news_items.append(item)
            if news_items:
                return "\n\n".join(news_items)
        except Exception as e:
            print(f"Error fetching yfinance news in get_news for {ticker}: {e}")
            
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
    print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Checking for breaking news...")
    seen_links, seen_titles = load_seen_news()
    new_articles = False
    
    for ticker, config in STOCKS.items():
        try:
            yf_ticker = config.get("yf_ticker", f"{ticker}.NS")
            t = yf.Ticker(yf_ticker)
            news = t.news
            if not news:
                continue
                
            for article in news[:3]:
                content = article.get('content', {})
                link = content.get('clickThroughUrl', {}).get('url') or content.get('canonicalUrl', {}).get('url') or ""
                title = content.get('title')
                pub_date_str = content.get('pubDate')
                
                if not link or not title or not pub_date_str:
                    continue
                    
                try:
                    pub_time = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except Exception as ex:
                    print(f"Error parsing date {pub_date_str}: {ex}")
                    continue
                    
                if pub_time <= STARTUP_TIME:
                    continue
                    
                if link not in seen_links and not is_duplicate_title(title, seen_titles):
                    seen_links.add(link)
                    seen_titles.add(title)
                    new_articles = True
                    
                    summary = content.get('summary') or content.get('description') or ""
                    sentiment = analyze_sentiment(title, summary)
                    color_emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
                    sentiment_label = "Positive" if sentiment == "positive" else "Negative" if sentiment == "negative" else "Neutral"
                    
                    summary_clean = re.sub('<[^<]+?>', '', summary).strip()
                    summary_short = summary_clean[:200] + "..." if len(summary_clean) > 200 else summary_clean
                    
                    msg = f"📰 <b>Breaking News: {config['name']}</b>\n\n"
                    msg += f"{color_emoji} <b>{title}</b>\n"
                    msg += f"<i>{summary_short}</i>\n\n"
                    msg += f"Sentiment: {color_emoji} {sentiment_label}\n"
                    msg += f"<a href='{link}'>🔗 Read Full Article</a>\n"
                    
                    send_telegram_message(msg)
                    print(f"Sent News Alert for {ticker}")
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            
    if new_articles:
        save_seen_news(seen_links, seen_titles)

def send_pre_market_report():
    print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Fetching pre-market report...")
    today = datetime.now(IST).date()
    if is_market_closed(today):
        send_telegram_message("🔔 <b>Notice:</b> Today market is closed.")
        print("Market is closed today. Sent holiday notice.")
        return
        
    query = "nifty+pre-market+OR+gift+nifty+moneycontrol+OR+economic+times"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        if feed.entries:
            best_entry = feed.entries[0]
            for entry in feed.entries[:5]:
                title_lower = entry.title.lower()
                if any(w in title_lower for w in ["open", "start", "gift nifty", "nifty"]):
                    best_entry = entry
                    break
            
            title = best_entry.title
            sentiment = analyze_sentiment(title, "")
            color_emoji = "🟢" if sentiment == "positive" else "🔴" if sentiment == "negative" else "🟡"
            
            msg = f"🔔 <b>Pre-Market Report (IST {datetime.now(IST).strftime('%H:%M')})</b>\n\n"
            msg += f"{color_emoji} {title}\n\n"
            msg += f"<a href='{best_entry.link}'>🔗 Read details on Google News</a>"
            
            send_telegram_message(msg)
            print("Sent pre-market report successfully.")
        else:
            print("No pre-market entries found.")
    except Exception as e:
        print(f"Error fetching pre-market report: {e}")

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
    global tv
    today = datetime.now(IST).date()
    if is_market_closed(today):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Today market is closed. Skipping stock analysis.")
        return
        
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running TradingView market analysis...")
    state = load_state()
    
    for ticker, config in STOCKS.items():
        try:
            # Fetch 200 bars of daily data from TradingView
            df = tv.get_hist(symbol=ticker, exchange=config['exchange'], interval=Interval.in_daily, n_bars=200)
            
            if df is None or df.empty:
                print(f"Warning: No data fetched for {ticker}. Re-initializing TradingView connection and retrying...")
                tv = TvDatafeed()
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
                        sell_reason = f"🛡️ <b>TRAILING STOP</b> Hit at ₹{current_close:.2f} (Locked in +{config['trail_buf']*100:.2f}% Buffer)"
                    else:
                        sell_reason = f"🛑 <b>STOP LOSS</b> Hit at ₹{current_close:.2f} (-{config['sl']*100:.2f}%)"
                        
                if sell_reason:
                    profit_pct = ((current_close - entry_price) / entry_price) * 100
                    
                    msg = f"━━━━━━━━━━━━━━━━━━━━━━\n📉 <b>SELL ALERT: {config['name']}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                    msg += f"🗓️ Date: {date_str}\n"
                    msg += f"💡 Reason: {sell_reason}\n\n"
                    msg += f"🚪 Entry Price: ₹{entry_price:.2f}\n"
                    msg += f"💵 Exit Price: ₹{current_close:.2f}\n"
                    msg += f"📊 Profit/Loss: <b>{profit_pct:.2f}%</b>\n\n"
                    prob = config.get('probability')
                    if prob is not None:
                        msg += f"🎲 Setup Probability: {prob * 100:.2f}%\n"
                    msg += f"\n📰 <b>Recent News:</b>\n{get_news(config['name'], ticker)}\n"
                    msg += f"\n#{ticker} #NSE"
                    
                    print(f"Sending SELL alert for {ticker} (Reason: {sell_reason})")
                    if send_telegram_message(msg):
                        add_reminder(ticker, "SELL", msg)
                        
                        # Log to closed trades
                        log_closed_trade(
                            ticker=ticker,
                            name=config['name'],
                            entry_price=entry_price,
                            exit_price=current_close,
                            entry_date=trade.get('date', 'Unknown'),
                            exit_date=date_str,
                            pnl_pct=profit_pct,
                            reason=sell_reason
                        )
                        
                        # Remove from active trades
                        del state[ticker]
                        save_state(state)
                    else:
                        print(f"Failed to send SELL alert for {ticker}. Will retry next cycle.")
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
                    msg = f"━━━━━━━━━━━━━━━━━━━━━━\n🚀 <b>BUY ALERT: {config['name']}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                    msg += f"🗓️ Date: {date_str}\n\n"
                    msg += f"🟢 Entry Price: ₹{current_close:.2f}\n"
                    msg += f"🎯 Target Price: ₹{current_close * (1 + config['tp']):.2f} (+{config['tp']*100:.2f}%)\n"
                    msg += f"🛡️ Stop Loss: ₹{current_close * (1 - config['sl']):.2f} (-{config['sl']*100:.2f}%)\n"
                    msg += f"📈 Trigger Price: +{config['trail_act']*100:.2f}%\n"
                    msg += f"🛡️ Trailing Stop-loss (Buffer): {config['trail_buf']*100:.2f}%\n"
                    prob = config.get('probability')
                    if prob is not None:
                        msg += f"🎲 Setup Probability: {prob * 100:.2f}%\n"
                    msg += f"\n📰 <b>Recent News:</b>\n{get_news(config['name'], ticker)}\n"
                    msg += f"\n#{ticker} #NSE"
                    
                    print(f"Sending BUY alert for {ticker}")
                    if send_telegram_message(msg):
                        # Enter Trade
                        state[ticker] = {
                            "entry_price": current_close,
                            "highest_price": current_high,
                            "date": date_str
                        }
                        save_state(state)
                        add_reminder(ticker, "BUY", msg)
                    else:
                        print(f"Failed to send BUY alert for {ticker}. Will retry next cycle.")
                else:
                    print(f"{ticker} [WAIT] - Cooldown: {is_cooled_off} ({hist_line:.2f}), Trend Intact: {is_trend_intact}")
                    
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

def run_scheduler():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Telegram Trading Bot Started!")
    send_telegram_message("🚀 <b>Trading Bot:</b> app updated and ready")
    
    # Run once immediately on startup
    analyze_stocks()
    check_news_stream()
    check_and_send_reminders()
    
    # Schedule trading check every hour
    schedule.every(1).hours.do(analyze_stocks)
    
    # Schedule breaking news check every 30 minutes
    schedule.every(30).minutes.do(check_news_stream)
    
    # Schedule pre-market report daily at 9:08 AM IST (Asia/Kolkata)
    schedule.every().day.at("09:08", "Asia/Kolkata").do(send_pre_market_report)
    
    # Schedule yesterday's reminders daily at 9:10 AM IST (Asia/Kolkata)
    schedule.every().day.at("09:10", "Asia/Kolkata").do(check_and_send_reminders)
    
    # Schedule weekly holdings report every Friday at 4:00 PM IST (Asia/Kolkata)
    schedule.every().friday.at("16:00", "Asia/Kolkata").do(send_holdings_report)
    
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
