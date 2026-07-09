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
import io
# pdfplumber is only needed for the earnings-surprise feature. Import it
# defensively so a missing dependency degrades that feature to "manual review"
# instead of breaking the whole bot at startup.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")
SEEN_NEWS_FILE = os.path.join(BASE_DIR, "seen_news.json")
SEEN_EARNINGS_FILE = os.path.join(BASE_DIR, "seen_earnings.json")
EARNINGS_ESTIMATES_FILE = os.path.join(BASE_DIR, "earnings_estimates.json")
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

def _keyword_sentiment(title, summary):
    """Fallback keyword sentiment (used only if FinBERT can't be loaded)."""
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


# FinBERT (ProsusAI/finbert) is a financial-domain sentiment model — far more
# accurate on market headlines than keyword matching. It's called via the free
# HuggingFace Inference API (no local model/torch needed). Set HF_API_TOKEN in .env
# to a free token from https://huggingface.co/settings/tokens. If the token is
# missing or the API errors, we fall back to keyword sentiment so the bot never breaks.
HF_API_TOKEN = clean_env_var(os.getenv("HF_API_TOKEN"))
HF_FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
_finbert_warned = False


def _finbert_api(text):
    """Query FinBERT via the HuggingFace Inference API. Returns a label or None."""
    global _finbert_warned
    if not HF_API_TOKEN:
        if not _finbert_warned:
            print("WARNING: HF_API_TOKEN not set — using keyword sentiment. "
                  "Add a free token from huggingface.co/settings/tokens to .env for FinBERT.")
            _finbert_warned = True
        return None
    try:
        resp = requests.post(
            HF_FINBERT_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": text[:2000], "options": {"wait_for_model": True}},
            timeout=20,
        )
        data = resp.json()
        # success shape: [[{"label": "...", "score": ...}, ...]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            best = max(data[0], key=lambda d: d.get("score", 0))
            return best["label"].lower()
        print(f"FinBERT API unexpected response ({data}); using keyword fallback.")
    except Exception as e:
        print(f"FinBERT API error ({e}); using keyword fallback.")
    return None


def analyze_sentiment(title, summary):
    """Return 'positive' / 'negative' / 'neutral' for a news item using FinBERT."""
    text = (title or "").strip()
    if summary:
        text = f"{text}. {summary.strip()}".strip()
    if not text:
        return "neutral"

    label = _finbert_api(text)
    if label in ("positive", "negative", "neutral"):
        return label
    return _keyword_sentiment(title, summary)

# Strategy Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
HIST_MIN = -10.0
HIST_MAX = 2.0
SMA_LEN = 50
SMA_PCT = 0.02 # Minimum 2% above 50 SMA

# Market-Crash Alert Configuration
# Fires a Telegram alert when the NIFTY 50 index falls this much (or more)
# from the previous close during the trading day.
NIFTY_INDEX_TICKER = "^NSEI"
MARKET_CRASH_PCT = -2.0

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

# ----------------------------------------------------------------------------
# Earnings-Surprise Alert Configuration
# ----------------------------------------------------------------------------
# NSE corporate-announcement "desc" (subject) values that indicate a results
# filing. Matched case-insensitively as substrings against the announcement's
# subject + attachment text. Tune this list to widen/narrow what we pick up.
EARNINGS_ANNOUNCEMENT_KEYWORDS = [
    "Financial Results",
    "Financial Result",
    "Board Meeting Outcome",
    "Outcome of Board Meeting",
    "Outcome of the Board Meeting",
]

# A metric is a "Beat" if the actual is more than this fraction above the
# estimate, a "Miss" if more than this fraction below, else "In-line".
EARNINGS_SURPRISE_THRESHOLD = 0.05  # 5%

# On startup / first run, only alert on filings this recent so we don't spam
# the whole historical announcement feed. Older filings are marked as seen
# (so they never re-alert) but do not generate an alert.
EARNINGS_LOOKBACK_DAYS = 2

# NSE endpoints + browser-like headers. The corporate-announcements API rejects
# requests that don't carry cookies from a prior page load, so we prime a
# session against the homepage first (see get_nse_session / fetch_nse_announcements).
NSE_BASE_URL = "https://www.nseindia.com"
NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    "Connection": "keep-alive",
}

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
        "EPL": {"exchange": "NSE", "name": "EPL", "tp": 0.27, "sl": 0.32, "trail_act": 0.18, "trail_buf": 0.09},
        "APOLLOHOSP": {"exchange": "NSE", "name": "Apollo Hospitals", "tp": 0.27, "sl": 0.23, "trail_act": 0.15, "trail_buf": 0.09},
        "BHARTIARTL": {"exchange": "NSE", "name": "Bharti Airtel", "tp": 0.30, "sl": 0.23, "trail_act": 0.14, "trail_buf": 0.07},
        "TORNTPOWER": {"exchange": "NSE", "name": "Torrent Power", "tp": 0.29, "sl": 0.23, "trail_act": 0.12, "trail_buf": 0.08},
        "PIDILITIND": {"exchange": "NSE", "name": "Pidilite", "tp": 0.28, "sl": 0.23, "trail_act": 0.13, "trail_buf": 0.07},
        "NATCOPHARM": {"exchange": "NSE", "name": "Natco Pharma", "tp": 0.27, "sl": 0.23, "trail_act": 0.13, "trail_buf": 0.09},
        "TVSMOTOR": {"exchange": "NSE", "name": "TVS Motors", "tp": 0.23, "sl": 0.26, "trail_act": 0.12, "trail_buf": 0.09},
        "BEL": {"exchange": "NSE", "name": "Bharat Electronics", "tp": 0.33, "sl": 0.24, "trail_act": 0.17, "trail_buf": 0.07},
        "GODREJCP": {"exchange": "NSE", "name": "Godrej Consumer Products", "tp": 0.23, "sl": 0.30, "trail_act": 0.19, "trail_buf": 0.16},
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
        "NIFTY": {"exchange": "NSE", "name": "NIFTY50 Index", "tp": 0.21, "sl": 0.27, "trail_act": 0.10, "trail_buf": 0.07, "yf_ticker": "^NSEI"},
        "KEI": {"exchange": "NSE", "name": "KEI Industries Limited", "tp": 0.26, "sl": 0.20, "trail_act": 0.15, "trail_buf": 0.05, "yf_ticker": "KEI.NS"},
        "NAVINFLUOR": {"exchange": "NSE", "name": "Navin Fluorine International Limited", "tp": 0.19, "sl": 0.185, "trail_act": 0.10, "trail_buf": 0.07, "yf_ticker": "NAVINFLUOR.NS"},
        "ZYDUSLIFE": {"exchange": "NSE", "name": "Zydus Lifesciences Limited", "tp": 0.18, "sl": 0.19, "trail_act": 0.08, "trail_buf": 0.06, "yf_ticker": "ZYDUSLIFE.NS"},
        "AJANTPHARM": {"exchange": "NSE", "name": "Ajanta Pharma", "tp": 0.21, "sl": 0.24, "trail_act": 0.09, "trail_buf": 0.08, "yf_ticker": "AJANTPHARM.NS"},
        "LUPIN": {"exchange": "NSE", "name": "Lupin Ltd", "tp": 0.15, "sl": 0.30, "trail_act": 0.12, "trail_buf": 0.11, "yf_ticker": "LUPIN.NS"},
        "RRKABEL": {"exchange": "NSE", "name": "RR Kabel Ltd", "tp": 0.10, "sl": 0.08, "trail_act": 0.08, "trail_buf": 0.06, "yf_ticker": "RRKABEL.NS"},
        "PRICOLLTD": {"exchange": "NSE", "name": "Pricol Ltd", "tp": 0.13, "sl": 0.17, "trail_act": 0.10, "trail_buf": 0.05, "yf_ticker": "PRICOLLTD.NS"},
        "THYROCARE": {"exchange": "NSE", "name": "Thyrocare", "tp": 0.15, "sl": 0.18, "trail_act": 0.08, "trail_buf": 0.04, "yf_ticker": "THYROCARE.NS"},
        "SJS": {"exchange": "NSE", "name": "SJS Enterprises", "tp": 0.25, "sl": 0.14, "trail_act": 0.13, "trail_buf": 0.11, "probability": 0.75, "yf_ticker": "SJS.NS"},
        "NH": {"exchange": "NSE", "name": "Narayana Hrudayalaya Ltd", "tp": 0.23, "sl": 0.26, "trail_act": 0.12, "trail_buf": 0.09, "probability": 0.92, "yf_ticker": "NH.NS"},
        "CAPLIPOINT": {"exchange": "NSE", "name": "Caplin Point Laboratories", "tp": 0.23, "sl": 0.26, "trail_act": 0.08, "trail_buf": 0.05, "probability": 0.81, "yf_ticker": "CAPLIPOINT.NS"},
        "MEDANTA": {"exchange": "NSE", "name": "Global Health Limited (Medanta)", "tp": 0.23, "sl": 0.18, "trail_act": 0.11, "trail_buf": 0.05, "probability": 0.85, "yf_ticker": "MEDANTA.NS"},
        "WAAREERTL": {"exchange": "NSE", "name": "Waaree Renewable Technologies Ltd", "tp": 0.28, "sl": 0.12, "trail_act": 0.10, "trail_buf": 0.08, "probability": 0.50, "yf_ticker": "WAAREERTL.NS"},
        "NEULANDLAB": {"exchange": "NSE", "name": "Neuland Laboratories", "tp": 0.26, "sl": 0.26, "trail_act": 0.10, "trail_buf": 0.07, "probability": 0.75, "yf_ticker": "NEULANDLAB.NS"},
        "GRSE": {"exchange": "NSE", "name": "Garden Reach Shipbuilders & Engineers Ltd", "tp": 0.19, "sl": 0.26, "trail_act": 0.14, "trail_buf": 0.11, "probability": 0.85, "yf_ticker": "GRSE.NS"},
        "SHAILY": {"exchange": "NSE", "name": "Shaily Engineering Plastics Limited", "tp": 0.16, "sl": 0.25, "trail_act": 0.10, "trail_buf": 0.07, "probability": 0.93, "yf_ticker": "SHAILY.NS"},
        "COALINDIA": {"exchange": "NSE", "name": "Coal India Ltd", "tp": 0.16, "sl": 0.20, "trail_act": 0.10, "trail_buf": 0.07, "probability": 1, "yf_ticker": "COALINDIA.NS"},
        "MCX": {"exchange": "NSE", "name": "Multi Commodity Exchange of India Limited", "tp": 0.18, "sl": 0.22, "trail_act": 0.06, "trail_buf": 0.04, "yf_ticker": "MCX.NS"},
        "MARUTI": {"exchange": "NSE", "name": "Maruti Suzuki India Limited", "tp": 0.24, "sl": 0.27, "trail_act": 0.13, "trail_buf": 0.11, "probability": 0.84, "yf_ticker": "MARUTI.NS"},
        "ASIANPAINT": {"exchange": "NSE", "name": "Asian Paints Ltd", "tp": 0.25, "sl": 0.28, "trail_act": 0.13, "trail_buf": 0.095, "probability": 0.9, "yf_ticker": "ASIANPAINT.NS"},
        "DRREDDY": {"exchange": "NSE", "name": "Dr Reddys Laboratories Ltd", "tp": 0.23, "sl": 0.29, "trail_act": 0.13, "trail_buf": 0.10, "probability": 0.84, "yf_ticker": "DRREDDY.NS"},
        "EICHERMOT": {"exchange": "NSE", "name": "Eicher Motors Limited", "tp": 0.20, "sl": 0.28, "trail_act": 0.16, "trail_buf": 0.13, "probability": 0.85, "yf_ticker": "EICHERMOT.NS"}
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

def load_seen_earnings():
    """Load the set of already-processed earnings announcement IDs."""
    seen = set()
    if os.path.exists(SEEN_EARNINGS_FILE):
        try:
            with open(SEEN_EARNINGS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    seen = set(data.get("processed", []))
                elif isinstance(data, list):
                    seen = set(data)
        except Exception as e:
            print(f"Error loading seen_earnings: {e}")
    return seen

def save_seen_earnings(seen):
    try:
        with open(SEEN_EARNINGS_FILE, "w") as f:
            json.dump({"processed": list(seen)}, f, indent=4)
    except Exception as e:
        print(f"Error saving seen_earnings: {e}")

def load_earnings_estimates():
    """Load analyst consensus estimates keyed by ticker -> quarter -> metrics.

    Schema (see earnings_estimates.json):
        {
          "BRITANNIA": {
            "Q1FY27": {"revenue": 4600, "net_profit": 560, "eps": 23.2}
          }
        }
    Revenue / net_profit are in ₹ crore; eps is ₹ per share. Keys starting
    with "_" (e.g. "_comment") are treated as documentation and ignored.
    """
    if os.path.exists(EARNINGS_ESTIMATES_FILE):
        try:
            with open(EARNINGS_ESTIMATES_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading earnings_estimates.json: {e}")
    return {}

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

MARKET_CRASH_STATE_FILE = os.path.join(BASE_DIR, "market_crash_state.json")

def load_market_crash_state():
    if os.path.exists(MARKET_CRASH_STATE_FILE):
        try:
            with open(MARKET_CRASH_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading market_crash_state.json: {e}")
    return {}

def save_market_crash_state(state):
    try:
        with open(MARKET_CRASH_STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Error saving market_crash_state.json: {e}")

def check_market_crash():
    """Alert once per trading day if the NIFTY 50 index falls MARKET_CRASH_PCT
    (or more) from the previous close. Independent of the per-stock strategy."""
    today = datetime.now(IST).date()
    if is_market_closed(today):
        return

    today_str = today.strftime("%Y-%m-%d")
    crash_state = load_market_crash_state()
    if crash_state.get("last_alert_date") == today_str:
        return  # Already alerted for today's drop

    try:
        nifty = yf.Ticker(NIFTY_INDEX_TICKER)
        fast_info = nifty.fast_info
        current_price = fast_info.get("last_price") or fast_info.get("lastPrice")
        prev_close = fast_info.get("previous_close") or fast_info.get("previousClose")

        if not current_price or not prev_close:
            print("[Market Crash] Could not fetch NIFTY price data.")
            return

        pct_change = ((current_price - prev_close) / prev_close) * 100

        if pct_change <= MARKET_CRASH_PCT:
            msg = f"━━━━━━━━━━━━━━━━━━━━━━\n🚨 <b>MARKET CRASH ALERT</b> 🚨\n━━━━━━━━━━━━━━━━━━━━━━\n"
            msg += f"📉 <b>NIFTY 50</b> is down <b>{pct_change:.2f}%</b> today\n\n"
            msg += f"🔹 Previous Close: ₹{prev_close:,.2f}\n"
            msg += f"🔹 Current Price: ₹{current_price:,.2f}\n\n"
            msg += f"⚠️ Sharp market-wide decline detected — review open positions.\n"
            msg += f"\n#NIFTY #NSE #MarketCrash"

            print(f"[Market Crash] NIFTY down {pct_change:.2f}%. Sending alert.")
            if send_telegram_message(msg):
                crash_state["last_alert_date"] = today_str
                save_market_crash_state(crash_state)
            else:
                print("[Market Crash] Failed to send alert. Will retry next cycle.")
        else:
            print(f"[Market Crash] NIFTY change: {pct_change:.2f}% (threshold {MARKET_CRASH_PCT:.2f}%)")
    except Exception as e:
        print(f"[Market Crash] Error checking NIFTY: {e}")

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
        
    msg_lines = ["━━━━━━━━━━━━━━━━━━━━━━\n📊 <b>Weekly Portfolio Report (Buy Signals - The Last 3 Months)</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"]
    
    # 1. Active Holdings
    msg_lines.append("🟩 <b>Active Holdings</b>")
    if not state:
        msg_lines.append("<i>No active holdings at the moment.</i>\n")
    else:
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
            
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            pnl_sign = "+" if pnl_pct >= 0 else ""
            
            msg_lines.append(
                f"• <b>{config.get('name', ticker)} ({ticker})</b>\n"
                f"  📅 Entry Date: {data['date']}\n"
                f"  🚪 Entry Price: ₹{entry_price:.2f}\n"
                f"  💵 Current Price: ₹{current_price:.2f}\n"
                f"  📊 P&L: {pnl_emoji} <b>{pnl_sign}{pnl_pct:.2f}%</b>\n"
            )
            
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

# ============================================================================
# Earnings-Surprise Alerts
# ============================================================================
# Standalone feature (independent of check_news_stream / get_news / analyze_stocks):
# polls NSE corporate-announcements for results filings by the stocks in STOCKS,
# extracts Revenue / Net Profit / EPS from the filing PDF, compares them to
# locally-maintained consensus estimates (earnings_estimates.json), and sends a
# numeric Beat/Miss/In-line surprise summary via Telegram/WhatsApp.

_nse_session = None

def get_nse_session():
    """Return a requests.Session primed with NSE cookies.

    NSE's JSON APIs 401/403 unless the session already holds cookies from a
    normal page load, so we hit the homepage + the filings page first.
    """
    global _nse_session
    if _nse_session is None:
        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        _nse_session = s
        _prime_nse_cookies(_nse_session)
    return _nse_session

def _prime_nse_cookies(session):
    try:
        session.get(NSE_BASE_URL, timeout=15)
        session.get(NSE_BASE_URL + "/companies-listing/corporate-filings-announcements", timeout=15)
    except Exception as e:
        print(f"[Earnings] Could not prime NSE cookies: {e}")

def nse_symbol_for(ticker, config):
    """Map a STOCKS key to its NSE trading symbol, or None if not an NSE equity.

    Uses yf_ticker (e.g. "BAJAJ-AUTO.NS" -> "BAJAJ-AUTO") when present; anything
    that isn't a ".NS" equity (e.g. the ^NSEI index) is skipped.
    """
    yf_ticker = config.get("yf_ticker")
    if yf_ticker:
        if yf_ticker.endswith(".NS"):
            return yf_ticker[:-3]
        return None  # index / non-NSE instrument — no corporate filings
    return ticker

def fetch_nse_announcements(symbol):
    """Fetch corporate announcements for a symbol. Returns a list, or None on
    a blocked/rate-limited/errored request (caller should skip this cycle)."""
    session = get_nse_session()
    url = NSE_ANNOUNCEMENTS_URL.format(symbol=requests.utils.quote(symbol))
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code in (401, 403):
                # Cookies likely expired — re-prime once and retry.
                print(f"[Earnings] NSE returned {resp.status_code} for {symbol}; refreshing cookies.")
                _prime_nse_cookies(session)
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            # Some responses wrap the rows in a dict.
            if isinstance(data, dict):
                return data.get("data") or data.get("rows") or []
            return []
        except Exception as e:
            print(f"[Earnings] Error fetching NSE announcements for {symbol}: {e}")
    return None

def _parse_nse_datetime(value):
    """Parse the assorted date formats NSE uses in announcement rows."""
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None

def derive_quarter_label(filing_date):
    """Best-effort mapping of a filing date to the fiscal quarter it reports.

    Indian FY runs Apr–Mar. Results are filed roughly one quarter in arrears:
      Jul–Sep -> Q1, Oct–Dec -> Q2, Jan–Mar -> Q3, Apr–Jun -> Q4 (annual).
    FY label is the year the fiscal year ends (e.g. Q1FY27 = Apr–Jun 2026).
    """
    if filing_date is None:
        return None
    m, y = filing_date.month, filing_date.year
    if 7 <= m <= 9:
        q, fy_end = 1, y + 1
    elif 10 <= m <= 12:
        q, fy_end = 2, y + 1
    elif 1 <= m <= 3:
        q, fy_end = 3, y
    else:  # 4–6
        q, fy_end = 4, y
    return f"Q{q}FY{fy_end % 100:02d}"

def get_consensus_estimate(ticker, quarter_label, estimates):
    """Return the estimate dict for a ticker/quarter, or None if unavailable."""
    tmap = estimates.get(ticker)
    if not isinstance(tmap, dict):
        return None
    qmap = {k: v for k, v in tmap.items() if not k.startswith("_")}
    if quarter_label and quarter_label in qmap:
        return qmap[quarter_label]
    # If the file holds exactly one quarter for this stock, use it regardless —
    # the derived quarter label is only a heuristic.
    if len(qmap) == 1:
        return next(iter(qmap.values()))
    return None

def _detect_unit_factor(text):
    """Return the multiplier to convert the PDF's stated units into ₹ crore."""
    head = text[:3000].lower()
    if "crore" in head or " cr" in head or "in cr" in head:
        return 1.0
    if "lakh" in head or "lac" in head:
        return 0.01   # 1 lakh = 0.01 crore
    if "million" in head or " mn" in head:
        return 0.1    # 1 million = 0.1 crore
    if "billion" in head or " bn" in head:
        return 100.0
    return 1.0  # default: assume crore (NSE large/mid caps usually file in crore)

def _extract_metric(text, labels, prefer_decimal=False):
    """Find the first numeric value following any of `labels` in `text`.

    NSE results tables list several columns (current quarter, prior quarter,
    year-ago quarter, ...); the first number after the row label is the current
    period. Values in parentheses are treated as negative. When prefer_decimal
    is set (EPS), a number containing a decimal point is preferred so we skip
    face-value noise like "equity share of Rs. 1 each".
    """
    low = text.lower()
    num_re = re.compile(r"(\()?\s*(-?[\d,]+(?:\.\d+)?)\s*(\)?)")
    for label in labels:
        idx = low.find(label.lower())
        if idx == -1:
            continue
        window = text[idx + len(label): idx + len(label) + 300]
        fallback = None
        for m in num_re.finditer(window):
            raw = m.group(2).replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            if m.group(1) and m.group(3):  # wrapped in ( ) -> negative
                val = -abs(val)
            if prefer_decimal and "." not in m.group(2):
                if fallback is None:
                    fallback = val
                continue
            return val
        if fallback is not None:
            return fallback
    return None

def parse_results_pdf(pdf_url):
    """Download and parse an NSE results PDF.

    Returns {"revenue", "net_profit", "eps"} (revenue/net_profit in ₹ crore,
    eps in ₹/share; any key may be None if not found), or None if the PDF can't
    be downloaded / opened at all.
    """
    if pdfplumber is None:
        print("[Earnings] pdfplumber not installed — cannot parse results PDF. "
              "Run: pip install pdfplumber")
        return None
    session = get_nse_session()
    try:
        resp = session.get(pdf_url, timeout=30)
        resp.raise_for_status()
        text_parts = []
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        text = "\n".join(text_parts)
    except Exception as e:
        print(f"[Earnings] Failed to download/parse PDF {pdf_url}: {e}")
        return None

    if not text.strip():
        print(f"[Earnings] No extractable text in PDF {pdf_url}")
        return None

    factor = _detect_unit_factor(text)
    revenue = _extract_metric(text, [
        "Revenue from operations", "Total income from operations",
        "Total revenue from operations", "Total Income",
    ])
    net_profit = _extract_metric(text, [
        "Net Profit/(Loss) for the period", "Net Profit / (Loss) for the period",
        "Profit/(loss) for the period", "Profit for the period",
        "Net Profit for the period", "Profit after tax", "Net Profit",
    ])
    eps = _extract_metric(text, [
        "Diluted (Rs", "Basic (Rs", "Basic & Diluted", "Basic and Diluted",
        "Basic EPS", "Earnings per share", "Earnings Per Share",
    ], prefer_decimal=True)

    return {
        "revenue": revenue * factor if revenue is not None else None,
        "net_profit": net_profit * factor if net_profit is not None else None,
        "eps": eps,  # per-share value, not scaled
    }

def _surprise(actual, estimate):
    """Return (surprise_pct, "Beat"/"Miss"/"In-line") or (None, None)."""
    if actual is None or estimate in (None, 0):
        return None, None
    pct = (actual - estimate) / estimate
    if pct > EARNINGS_SURPRISE_THRESHOLD:
        label = "Beat"
    elif pct < -EARNINGS_SURPRISE_THRESHOLD:
        label = "Miss"
    else:
        label = "In-line"
    return pct, label

def _metric_line(display_name, actual, estimate, is_eps=False):
    """Build one 'Metric: actual vs est (pct — label)' line, or a raw/N-A line."""
    def fmt(v):
        if v is None:
            return "N/A"
        return f"₹{v:.2f}" if is_eps else f"₹{v:,.0f}cr"
    pct, label = _surprise(actual, estimate)
    if pct is None:
        # Not enough to compute a surprise — show whatever we have.
        return f"<b>{display_name}:</b> {fmt(actual)} vs est {fmt(estimate)}"
    return (f"<b>{display_name}:</b> {fmt(actual)} vs est {fmt(estimate)} "
            f"({pct*100:+.1f}% — {label})")

def _overall_label(labels):
    labels = [l for l in labels if l]
    if not labels:
        return None
    beats, misses = labels.count("Beat"), labels.count("Miss")
    if beats > misses:
        return "Beat"
    if misses > beats:
        return "Miss"
    return "In-line"

def check_earnings_surprises():
    """Poll NSE for new results filings by our stocks and alert on the surprise
    vs consensus. Independent of the news/analysis functions; own schedule."""
    try:
        today = datetime.now(IST).date()
        if is_market_closed(today):
            print(f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Market closed. Skipping earnings check.")
            return

        print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Checking for earnings-surprise filings...")
        processed = load_seen_earnings()
        estimates = load_earnings_estimates()
        changed = False

        for ticker, config in STOCKS.items():
            symbol = nse_symbol_for(ticker, config)
            if not symbol:
                continue

            announcements = fetch_nse_announcements(symbol)
            if announcements is None:
                # NSE blocked/errored for this symbol — skip it this cycle.
                continue

            for ann in announcements[:15]:
                desc = (ann.get("desc") or ann.get("attchmntText") or "") + " " + (ann.get("attchmntText") or "")
                if not any(kw.lower() in desc.lower() for kw in EARNINGS_ANNOUNCEMENT_KEYWORDS):
                    continue

                pdf_url = ann.get("attchmntFile") or ""
                ann_id = pdf_url or f"{symbol}|{ann.get('an_dt') or ann.get('sort_date')}"
                if not ann_id or ann_id in processed:
                    continue

                filed_dt = _parse_nse_datetime(ann.get("an_dt") or ann.get("sort_date"))
                filed_date = filed_dt.date() if filed_dt else today
                filed_str = filed_dt.strftime("%d %b %Y, %H:%M") if filed_dt else "Unknown"

                # Seed old filings as seen without alerting (avoids first-run spam).
                if (today - filed_date).days > EARNINGS_LOOKBACK_DAYS:
                    processed.add(ann_id)
                    changed = True
                    continue

                print(f"[Earnings] New results filing for {ticker} ({symbol}) filed {filed_str}")

                actuals = parse_results_pdf(pdf_url) if pdf_url.lower().endswith(".pdf") else None
                quarter_label = derive_quarter_label(filed_dt)
                est = get_consensus_estimate(ticker, quarter_label, estimates)

                # If we can't parse the filing OR have no estimate, send the
                # simpler "manual review needed" alert (with any parsed actuals).
                if actuals is None or est is None:
                    msg = _build_manual_review_alert(ticker, config, filed_str, quarter_label, actuals, est, pdf_url)
                else:
                    msg = _build_surprise_alert(ticker, config, filed_str, quarter_label, actuals, est, pdf_url)

                if send_telegram_message(msg):
                    processed.add(ann_id)
                    changed = True
                    print(f"[Earnings] Sent earnings alert for {ticker}")
                else:
                    print(f"[Earnings] Failed to send earnings alert for {ticker}; will retry next cycle.")

            time.sleep(1)  # be gentle with NSE between symbols

        if changed:
            save_seen_earnings(processed)
    except Exception as e:
        # Never let this crash the scheduler loop.
        print(f"[Earnings] Unexpected error in check_earnings_surprises: {e}")

def _build_surprise_alert(ticker, config, filed_str, quarter_label, actuals, est, pdf_url):
    lines = _metric_lines_and_labels(actuals, est)
    overall = _overall_label(lines["labels"])
    overall_emoji = {"Beat": "🟢🚀", "Miss": "🔴", "In-line": "🟡"}.get(overall, "🟡")

    q = f" ({quarter_label})" if quarter_label else ""
    msg = f"━━━━━━━━━━━━━━━━━━━━━━\n📊 <b>EARNINGS ALERT: {config['name']}</b>{q}\n━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🗓️ Filed: {filed_str}\n\n"
    msg += lines["revenue"] + "\n"
    msg += lines["net_profit"] + "\n"
    msg += lines["eps"] + "\n\n"
    if overall:
        msg += f"{overall_emoji} <b>Overall: {overall}</b>\n"
    if pdf_url:
        msg += f"<a href='{pdf_url}'>🔗 View Filing</a>\n"
    msg += f"\n#{ticker} #NSE #Earnings"
    return msg

def _metric_lines_and_labels(actuals, est):
    est = est or {}
    rev_line = _metric_line("Revenue", actuals.get("revenue"), est.get("revenue"))
    np_line = _metric_line("Net Profit", actuals.get("net_profit"), est.get("net_profit"))
    eps_line = _metric_line("EPS", actuals.get("eps"), est.get("eps"), is_eps=True)
    labels = [
        _surprise(actuals.get("revenue"), est.get("revenue"))[1],
        _surprise(actuals.get("net_profit"), est.get("net_profit"))[1],
        _surprise(actuals.get("eps"), est.get("eps"))[1],
    ]
    return {"revenue": rev_line, "net_profit": np_line, "eps": eps_line, "labels": labels}

def _build_manual_review_alert(ticker, config, filed_str, quarter_label, actuals, est, pdf_url):
    q = f" ({quarter_label})" if quarter_label else ""
    msg = f"━━━━━━━━━━━━━━━━━━━━━━\n📊 <b>EARNINGS ALERT: {config['name']}</b>{q}\n━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🗓️ Filed: {filed_str}\n\n"
    if actuals:
        # We parsed something but couldn't compute a surprise (no estimate).
        lines = _metric_lines_and_labels(actuals, est)
        msg += lines["revenue"] + "\n"
        msg += lines["net_profit"] + "\n"
        msg += lines["eps"] + "\n\n"
    reason = "estimate data unavailable" if actuals else "could not parse the filing PDF"
    msg += f"⚠️ <b>Manual review needed</b> — could not compute surprise ({reason}).\n"
    if pdf_url:
        msg += f"<a href='{pdf_url}'>🔗 View Filing</a>\n"
    msg += f"\n#{ticker} #NSE #Earnings"
    return msg

# ============================================================================
# Pre-Market Brief
# ============================================================================
# Global cues, commodities, FII/DII flows, VIX, and the prior session's close
# + breadth, all from directly-callable sources (yfinance + NSE JSON APIs) so
# this can run unattended on a schedule — no MCP/interactive tools involved.
#
# GIFT Nifty has no free structured API (checked yfinance, NSE's public
# endpoints, and tapetide's MCP tools) — it's text-mined from Google News
# headlines instead (see fetch_gift_nifty_setup). Best-effort: only used when
# a fresh, on-topic headline actually contains a parseable points figure; we
# omit the line rather than fabricate a number or show stale news.
#
# Breadth is Nifty-50-scoped (NSE has no public full-market breadth endpoint),
# not full-market — labelled as such below so it isn't misread as the latter.

GIFT_NIFTY_LOOKBACK_HOURS = 12
GIFT_NIFTY_POINTS_RE = re.compile(
    r'\b(up|down|higher|lower|gains?|falls?|jumps?|slips?|dips?|surges?|plunges?|advances?|declines?)\b'
    r'.{0,20}?(\d+(?:\.\d+)?)\s*pts?\b',
    re.IGNORECASE,
)
GIFT_NIFTY_POSITIVE_WORDS = {'up', 'higher', 'gain', 'jump', 'surge', 'advance'}
GIFT_NIFTY_NEGATIVE_WORDS = {'down', 'lower', 'fall', 'slip', 'dip', 'plunge', 'decline'}

def fetch_gift_nifty_setup():
    """Best-effort GIFT Nifty pre-open signal, text-mined from recent Google
    News headlines (no free structured API exists for GIFT Nifty). Returns an
    HTML-formatted line, or None if nothing fresh/on-topic is found — we'd
    rather omit the line than show a stale or fabricated number."""
    url = "https://news.google.com/rss/search?q=gift+nifty&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"[Pre-Market] Error fetching GIFT Nifty news: {e}")
        return None

    now_utc = datetime.now(timezone.utc)
    fallback_entry = None

    for entry in feed.entries[:15]:
        title = entry.get('title', '')
        if 'gift nifty' not in title.lower():
            continue

        pub = entry.get('published_parsed')
        if not pub:
            continue
        pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
        if (now_utc - pub_dt).total_seconds() > GIFT_NIFTY_LOOKBACK_HOURS * 3600:
            continue

        if fallback_entry is None:
            fallback_entry = entry

        match = GIFT_NIFTY_POINTS_RE.search(title)
        if not match:
            continue

        direction_word = match.group(1).lower().rstrip('s')
        if direction_word in GIFT_NIFTY_POSITIVE_WORDS:
            sign, bias = '+', 'positive'
        elif direction_word in GIFT_NIFTY_NEGATIVE_WORDS:
            sign, bias = '-', 'negative'
        else:
            continue

        points = float(match.group(2))
        magnitude = "mildly " if points < 20 else ("sharply " if points >= 60 else "")
        return f"🌐 <b>Setup:</b> GIFT Nifty {sign}{points:.0f} pts → {magnitude}{bias} open"

    if fallback_entry is not None:
        return f"🌐 <b>Setup:</b> {fallback_entry.title} <i>(no exact point move found)</i>"

    return None

GLOBAL_CUES_TICKERS = {
    "Nikkei": "^N225",
    "Hang Seng": "^HSI",
    "Dow": "^DJI",
    "Nasdaq": "^IXIC",
}
COMMODITY_TICKERS = {
    "Crude": "CL=F",
    "Gold": "GC=F",
}
USD_INR_TICKER = "INR=X"
INDIA_VIX_TICKER = "^INDIAVIX"

NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
NSE_ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"
FII_DII_HISTORY_FILE = os.path.join(BASE_DIR, "fii_dii_history.json")

def _yf_change(ticker):
    """Return (last_price, pct_change_vs_previous_close), or (None, None) on failure."""
    try:
        fi = yf.Ticker(ticker).fast_info
        last = fi.get("last_price") or fi.get("lastPrice")
        prev = fi.get("previous_close") or fi.get("previousClose")
        if not last or not prev:
            return None, None
        return last, ((last - prev) / prev) * 100
    except Exception as e:
        print(f"[Pre-Market] Error fetching {ticker}: {e}")
        return None, None

def _format_pct(pct, flat_threshold=0.1):
    if pct is None:
        return "N/A"
    if abs(pct) < flat_threshold:
        return "flat"
    return f"{pct:+.1f}%"

def fetch_fii_dii_flows():
    """Return (fii_net_cr, dii_net_cr, date_str) for the latest reported session, or (None, None, None)."""
    try:
        session = get_nse_session()
        resp = session.get(NSE_FII_DII_URL, timeout=20)
        if resp.status_code in (401, 403):
            _prime_nse_cookies(session)
            resp = session.get(NSE_FII_DII_URL, timeout=20)
        resp.raise_for_status()
        rows = resp.json()
        fii_net = dii_net = date_str = None
        for row in rows:
            category = (row.get("category") or "").upper()
            if "FII" in category:
                fii_net = float(row.get("netValue"))
                date_str = row.get("date")
            elif "DII" in category:
                dii_net = float(row.get("netValue"))
                date_str = date_str or row.get("date")
        return fii_net, dii_net, date_str
    except Exception as e:
        print(f"[Pre-Market] Error fetching FII/DII flows: {e}")
        return None, None, None

def fetch_nifty50_breadth():
    """Return (advances, declines, unchanged) for the Nifty 50 constituents, or (None, None, None)."""
    try:
        session = get_nse_session()
        resp = session.get(NSE_ALL_INDICES_URL, timeout=20)
        if resp.status_code in (401, 403):
            _prime_nse_cookies(session)
            resp = session.get(NSE_ALL_INDICES_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        nifty = next((d for d in data if d.get("indexSymbol") == "NIFTY 50"), None)
        if not nifty:
            return None, None, None
        return int(nifty["advances"]), int(nifty["declines"]), int(nifty["unchanged"])
    except Exception as e:
        print(f"[Pre-Market] Error fetching Nifty breadth: {e}")
        return None, None, None

def record_fii_streak(flow_date, fii_net):
    """Persist today's FII net flow and return the current same-direction streak length."""
    history = {}
    if os.path.exists(FII_DII_HISTORY_FILE):
        try:
            with open(FII_DII_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception as e:
            print(f"[Pre-Market] Error loading fii_dii_history.json: {e}")

    if flow_date and fii_net is not None:
        history[flow_date] = fii_net
        try:
            with open(FII_DII_HISTORY_FILE, "w") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"[Pre-Market] Error saving fii_dii_history.json: {e}")

    if not history:
        return 1
    dates_sorted = sorted(history.keys(), reverse=True)
    sign = 1 if history[dates_sorted[0]] >= 0 else -1
    streak = 0
    for d in dates_sorted:
        day_sign = 1 if history[d] >= 0 else -1
        if day_sign != sign:
            break
        streak += 1
    return streak

def send_pre_market_report():
    print(f"\n[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] Fetching pre-market report...")
    today = datetime.now(IST).date()
    if is_market_closed(today):
        send_telegram_message("🔔 <b>Notice:</b> Today market is closed.")
        print("Market is closed today. Sent holiday notice.")
        return

    lines = [f"📊 <b>Pre-Market Brief — {datetime.now(IST).strftime('%a, %d %b')}</b>"]

    gift_nifty_line = fetch_gift_nifty_setup()
    if gift_nifty_line:
        lines.append(gift_nifty_line)

    # Global cues
    cue_parts = []
    for name, ticker in GLOBAL_CUES_TICKERS.items():
        _, pct = _yf_change(ticker)
        cue_parts.append(f"{name} {_format_pct(pct)}")
    lines.append(f"🌏 <b>Global cues:</b> {', '.join(cue_parts)}")

    # Commodities + USD/INR
    commodity_parts = []
    for name, ticker in COMMODITY_TICKERS.items():
        _, pct = _yf_change(ticker)
        commodity_parts.append(f"{name} {_format_pct(pct)}")
    usdinr_price, _ = _yf_change(USD_INR_TICKER)
    usdinr_str = f"{usdinr_price:.2f}" if usdinr_price else "N/A"
    lines.append(f"🛢️ <b>Commodities:</b> {', '.join(commodity_parts)} | USD/INR {usdinr_str}")

    # FII/DII flows (with a real, persisted FII streak — not guessed)
    fii_net, dii_net, flow_date = fetch_fii_dii_flows()
    if fii_net is not None and dii_net is not None:
        streak = record_fii_streak(flow_date, fii_net)
        streak_word = "buy" if fii_net >= 0 else "sell"
        streak_note = f" ({streak}-day {streak_word} streak)" if streak > 1 else ""
        fii_sign = "+" if fii_net >= 0 else ""
        dii_sign = "+" if dii_net >= 0 else ""
        lines.append(
            f"💰 <b>Flows ({flow_date}):</b> FII {fii_sign}₹{fii_net:,.0f} Cr{streak_note} | "
            f"DII {dii_sign}₹{dii_net:,.0f} Cr"
        )
    else:
        lines.append("💰 <b>Flows:</b> data unavailable")

    # India VIX
    vix_level, vix_pct = _yf_change(INDIA_VIX_TICKER)
    if vix_level is not None:
        note = ""
        if vix_pct >= 15:
            note = " — a real jump, worth noting on position sizing"
        elif vix_pct <= -15:
            note = " — a sharp drop in fear"
        lines.append(f"😨 <b>VIX:</b> {vix_level:.2f} ({_format_pct(vix_pct)}{note})")
    else:
        lines.append("😨 <b>VIX:</b> data unavailable")

    # Yesterday's Nifty close + Nifty-50 breadth
    nifty_level, nifty_pct = _yf_change(NIFTY_INDEX_TICKER)
    adv, dec, unch = fetch_nifty50_breadth()
    if nifty_level is not None:
        close_line = f"📉 <b>Yesterday's close:</b> Nifty {nifty_level:,.0f} ({_format_pct(nifty_pct)})"
        if adv is not None:
            close_line += f", Nifty-50 breadth {adv} adv / {dec} dec"
        lines.append(close_line)
    else:
        lines.append("📉 <b>Yesterday's close:</b> data unavailable")

    # Overall sentiment indicator (Nifty direction + VIX move + FII flow)
    score = 0
    if nifty_pct is not None:
        score += 1 if nifty_pct > 0 else -1 if nifty_pct < 0 else 0
    if vix_pct is not None:
        score += -1 if vix_pct > 5 else (1 if vix_pct < -5 else 0)
    if fii_net is not None:
        score += 1 if fii_net > 0 else -1 if fii_net < 0 else 0

    if score >= 2:
        sentiment_emoji, sentiment_label = "🟢", "Positive bias"
    elif score <= -2:
        sentiment_emoji, sentiment_label = "🔴", "Cautious / negative bias"
    else:
        sentiment_emoji, sentiment_label = "🟡", "Mixed / neutral bias"
    lines.append(f"\n{sentiment_emoji} <b>Sentiment: {sentiment_label}</b>")

    msg = "\n".join(lines)
    if send_telegram_message(msg):
        print("Sent pre-market report successfully.")
    else:
        print("Failed to send pre-market report.")

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
    check_earnings_surprises()
    check_and_send_reminders()
    check_market_crash()

    # Schedule trading check every hour
    schedule.every(1).hours.do(analyze_stocks)

    # Schedule breaking news check every 30 minutes
    schedule.every(30).minutes.do(check_news_stream)

    # Schedule market-crash check every 5 minutes (needs to catch sudden drops fast)
    schedule.every(5).minutes.do(check_market_crash)

    # Schedule earnings-surprise filing check every 15 minutes (to catch
    # results filings quickly during market hours)
    schedule.every(15).minutes.do(check_earnings_surprises)
    
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
