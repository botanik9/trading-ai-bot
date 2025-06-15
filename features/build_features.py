import os
import pandas as pd
from technical_indicators import add_technical_indicators

RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
SYMBOL = "BTCUSDT"
TIMEFRAMES = ["1d", "4h", "1h", "15m"]

def load_raw_data(symbol, timeframe):
    filepath = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_1year.csv")
    df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])
    return df

def prepare_features():
    dfs = []
    for tf in TIMEFRAMES:
        df = load_raw_data(SYMBOL, tf)
        df.set_index("open_time", inplace=True)
        df = add_technical_indicators(df, tf)
        # Переименовываем колонки с суффиксом таймфрейма для объединения
        df = df.add_suffix(f"_{tf}")
        dfs.append(df)
    # Объединяем все таймфреймы по индексу open_time
    combined = pd.concat(dfs, axis=1, join="outer")
    combined.sort_index(inplace=True)
    # Сохраняем в processed
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    combined.to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_features.csv"))
    print("Processed features saved.")

if __name__ == "__main__":
    prepare_features()
