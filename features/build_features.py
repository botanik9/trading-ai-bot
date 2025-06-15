import pandas as pd
import os
from features.technical_indicators import add_technical_indicators

BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/processed_features.csv")

timeframes = ["1d", "4h", "1h", "15m"]

dfs = []

for tf in timeframes:
    filename = f"BTCUSDT_{tf}_1year.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    # Загрузка CSV
    df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])
    
    # Установка времени открытия как индекс
    df.set_index("open_time", inplace=True)
    
    # Добавление технических индикаторов
    df = add_technical_indicators(df, tf)
    
    # Переименование колонок с суффиксом таймфрейма
    df = df.add_suffix(f"_{tf}")
    
    # Переименование индекса
    df.index.name = "open_time"
    
    dfs.append(df)

# Объединение по пересечению временных меток
combined = dfs[0]
for df in dfs[1:]:
    combined = combined.join(df, how="inner")

# Сортировка по времени
combined.sort_index(inplace=True)

# Сохранение итогового файла
combined.to_csv(PROCESSED_DATA_PATH)
print(f"Синхронизированные фичи сохранены в: {PROCESSED_DATA_PATH}")

