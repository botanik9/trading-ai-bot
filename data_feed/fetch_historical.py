import os
import requests
import time
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = "data/raw"
SYMBOL = "BTCUSDT"
INTERVALS = ["1d", "4h", "1h", "15m"]
LIMIT = 1000

def fetch_klines(symbol, interval, start_ts, end_ts):
    klines = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": LIMIT
        }
        resp = requests.get(BASE_URL, params=params)
        data = resp.json()
        if not data:
            break
        klines.extend(data)
        last_ts = data[-1][0]
        if last_ts == start_ts:
            # Предотвращаем зацикливание
            break
        start_ts = last_ts + 1
        time.sleep(0.3)
    return klines

def save_klines_to_csv(klines, filename):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df.to_csv(filename, index=False)

def main():
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=365)
    os.makedirs(DATA_DIR, exist_ok=True)

    for interval in INTERVALS:
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        print(f"Fetching {SYMBOL} {interval} klines from {start_dt} to {end_dt}")

        klines = fetch_klines(SYMBOL, interval, start_ts, end_ts)
        filename = os.path.join(DATA_DIR, f"{SYMBOL}_{interval}_1year.csv")
        save_klines_to_csv(klines, filename)
        print(f"Saved {len(klines)} records to {filename}")

if __name__ == "__main__":
    main()

