from tvDatafeed import TvDatafeed, Interval
import sys
try:
    tv = TvDatafeed()
    df = tv.get_hist(symbol='BRITANNIA', exchange='NSE', interval=Interval.in_daily, n_bars=10)
    print("Success:")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
