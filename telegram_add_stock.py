import os
import json
import time
import requests
import itertools
from datetime import datetime
from dotenv import load_dotenv
from tvDatafeed import TvDatafeed, Interval

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
STOCKS_FILE = os.path.join(BASE_DIR, "stocks_config.json")

# Load env variables
load_dotenv(ENV_FILE)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Strategy Parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9
HIST_MIN = -10.0
HIST_MAX = 2.0
SMA_LEN = 50
SMA_PCT = 0.02

# Grid Search Parameters
TP_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SL_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25]
TRAIL_ACT_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TRAIL_BUF_RANGE = [0.02, 0.05, 0.08, 0.10, 0.15]

tv = TvDatafeed()

def calculate_indicators(df):
    if df is None or len(df) < SMA_LEN:
        return None
    df['SMA_50'] = df['close'].rolling(window=SMA_LEN).mean()
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIG, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def simulate_strategy(df, tp, sl, trail_act, trail_buf):
    trades = []
    in_trade = False
    entry_price = 0
    highest_price = 0
    target_price = 0
    activation_price = 0
    stop_loss_price = 0
    trailing_stop_price = 0
    
    # Pre-compute conditions
    hist = df['MACD_Hist'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    sma = df['SMA_50'].values
    
    for i in range(len(df)):
        if i < SMA_LEN:
            continue
            
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        
        if in_trade:
            # First check if stop loss is hit (conservative exit)
            if current_low <= stop_price:
                profit = (stop_price - entry_price) / entry_price
                trades.append(profit)
                in_trade = False
                continue
                
            # Next check if target is hit
            if current_high >= target_price:
                profit = (target_price - entry_price) / entry_price
                trades.append(profit)
                in_trade = False
                continue

            # Update trailing stop if we survived the day
            if current_high > highest_price:
                highest_price = current_high
            
            if highest_price >= activation_price:
                stop_price = trailing_stop_price
        else:
            # Entry check
            if (HIST_MIN < hist[i] <= HIST_MAX) and (current_close > sma[i] * (1 + SMA_PCT)):
                in_trade = True
                entry_price = current_close
                highest_price = current_high
                
                target_price = entry_price * (1 + tp)
                activation_price = entry_price * (1 + trail_act)
                stop_loss_price = entry_price * (1 - sl)
                trailing_stop_price = entry_price * (1 + trail_buf)
                stop_price = stop_loss_price
                
    return trades

def run_backtest(ticker, chat_id):
    send_message(chat_id, f"🔍 Fetching historical data for {ticker} (max bars)...")
    global tv
    df = tv.get_hist(symbol=ticker, exchange='NSE', interval=Interval.in_daily, n_bars=10000)
    if df is None or df.empty:
        # Retry once
        tv = TvDatafeed()
        df = tv.get_hist(symbol=ticker, exchange='NSE', interval=Interval.in_daily, n_bars=10000)
        
    if df is None or df.empty:
        send_message(chat_id, f"❌ Failed to fetch TradingView data for {ticker}. Check symbol name.")
        return False
        
    df = calculate_indicators(df)
    if df is None:
        send_message(chat_id, f"❌ Not enough data for {ticker} to calculate indicators.")
        return False
        
    send_message(chat_id, f"🧪 Running parameter grid search on {len(df)} days of data. This might take a minute...")
    
    valid_results = []
    combinations = list(itertools.product(TP_RANGE, SL_RANGE, TRAIL_ACT_RANGE, TRAIL_BUF_RANGE))
    
    for tp, sl, trail_act, trail_buf in combinations:
        # Invalid config check (trail_act must be >= trail_buf + small margin)
        if trail_act <= trail_buf:
            continue
            
        trades = simulate_strategy(df, tp, sl, trail_act, trail_buf)
        if not trades:
            continue
            
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        win_rate = len(wins) / len(trades)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
        
        if win_rate > 0.60 and profit_factor > 3.0:
            valid_results.append({
                'tp': tp, 'sl': sl, 'trail_act': trail_act, 'trail_buf': trail_buf,
                'win_rate': win_rate, 'profit_factor': profit_factor, 'total_trades': len(trades)
            })
            
    if not valid_results:
        send_message(chat_id, f"⚠️ Backtest finished.\nCould not find any parameters for {ticker} that satisfy Win Rate > 60% AND Profit Factor > 3.0.\nTry adding manually.")
        return False
        
    # Sort by profit factor descending
    valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)
    best = valid_results[0]
    
    msg = (f"✅ <b>Optimization Complete for {ticker}</b>\n\n"
           f"<b>Best Parameters:</b>\n"
           f"Take Profit (tp): {best['tp']}\n"
           f"Stop Loss (sl): {best['sl']}\n"
           f"Trail Activation: {best['trail_act']}\n"
           f"Trail Buffer: {best['trail_buf']}\n\n"
           f"<b>Performance:</b>\n"
           f"Win Rate: {best['win_rate']*100:.1f}%\n"
           f"Profit Factor: {best['profit_factor']:.2f}\n"
           f"Total Trades: {best['total_trades']}\n\n"
           f"Adding to configuration...")
           
    send_message(chat_id, msg)
    return best

def add_stock_to_config(ticker, best_params):
    stocks = {}
    if os.path.exists(STOCKS_FILE):
        try:
            with open(STOCKS_FILE, "r") as f:
                stocks = json.load(f)
        except json.JSONDecodeError:
            stocks = {}
            
    stocks[ticker] = {
        "exchange": "NSE",
        "name": ticker,
        "tp": best_params['tp'],
        "sl": best_params['sl'],
        "trail_act": best_params['trail_act'],
        "trail_buf": best_params['trail_buf']
    }
    
    with open(STOCKS_FILE, "w") as f:
        json.dump(stocks, f, indent=4)

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error sending message: {e}")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN is not set in .env file.")
        return
        
    print("Starting Telegram listener for adding stocks...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    offset = None
    
    while True:
        try:
            params = {"timeout": 60}
            if offset:
                params["offset"] = offset
                
            response = requests.get(url, params=params, timeout=70)
            if response.status_code == 200:
                data = response.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if not message:
                        continue
                        
                    text = message.get("text", "").strip()
                    chat_id = message["chat"]["id"]
                    
                    if text.startswith("/addstock"):
                        parts = text.split()
                        if len(parts) < 2:
                            send_message(chat_id, "Usage: /addstock <SYMBOL>\nExample: /addstock RELIANCE")
                            continue
                            
                        ticker = parts[1].upper()
                        best_params = run_backtest(ticker, chat_id)
                        if best_params:
                            add_stock_to_config(ticker, best_params)
                            send_message(chat_id, f"🎉 Successfully added {ticker} to stocks_config.json!\nThe main bot will pick it up automatically.")
                            
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Error in polling loop: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
