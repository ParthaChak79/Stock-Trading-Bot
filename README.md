# Telegram Trading Bot - AI Wealth Builder Strategy

This Python application monitors your preferred NSE stocks and sends BUY and SELL alerts to your Telegram account, based on the **AI Wealth Builder Strategy**. 

The bot implements MACD Cooldown and SMA 50 Trend filtering for entry, and custom Trailing Stop configurations per stock for exit. It also fetches the top recent news using Google News RSS.

## Setup Instructions

1. **Install Python Dependencies:**
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Your Telegram Bot:**
   - Open Telegram and search for **@BotFather**.
   - Send `/newbot` and follow the prompts to create a new bot.
   - BotFather will give you an **API Token**. Keep this secure.
   - Start a conversation with your new bot on Telegram (press Start).

3. **Get Your Telegram Chat ID:**
   - Search for **@userinfobot** or **@RawDataBot** on Telegram and start it.
   - It will reply with your `Id` (a number like `123456789`). This is your Chat ID.

4. **Configure Environment Variables:**
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` in a text editor and add your Bot Token and Chat ID.

## Running the Bot

Run the application:
```bash
python app.py
```

- When the bot runs, it will immediately perform an analysis of your configured stocks.
- After the initial check, it is scheduled to run every 1 hour while the script is active.
- If a BUY or SELL signal is triggered, it will immediately notify you on Telegram along with top news!
- Active trades are saved in a local `portfolio_state.json` file. This allows the bot to remember your open positions and trailing stops even if you restart it.
