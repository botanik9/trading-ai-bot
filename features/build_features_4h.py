import pandas as pd
import os
from technical_indicators import (
    add_technical_indicators,
    find_order_blocks,
    find_fvg,
    find_liquidity_levels,
    add_order_block_feature,
    add_fvg_feature,
    add_liquidity_feature
)

BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/processed_features.csv")

# Загружаем и обрабатываем данные для 4h
tf = "4h"
filename = f"BTCUSDT_{tf}_3year.csv"
filepath = os.path.join(RAW_DATA_DIR, filename)
df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])
df.set_index("open_time", inplace=True)
df = add_technical_indicators(df, tf)

# Находим order blocks, FVG и уровни ликвидности на соответствующих таймфреймах
order_blocks_4h = find_order_blocks(df)  # Order blocks на том же таймфрейме
fvg_4h = find_fvg(df)  # FVG на том же таймфрейме
liquidity_1d = find_liquidity_levels(df)  # Ликвидность можно искать на более старшем TF

# Добавляем новые признаки
df = add_order_block_feature(df, order_blocks_4h)
df = add_fvg_feature(df, fvg_4h)
df = add_liquidity_feature(df, liquidity_1d)

df.index.name = "open_time"

df.to_csv(PROCESSED_DATA_PATH)
print(f"Processed features for 4h saved to {PROCESSED_DATA_PATH}")
