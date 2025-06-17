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

timeframes = ["1d", "4h", "1h", "15m"]

dfs = {}

# Загружаем и обрабатываем старшие таймфреймы (1d, 4h, 1h)
for tf in ["1d", "4h", "1h"]:
    filename = f"BTCUSDT_{tf}_3year.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])
    df.set_index("open_time", inplace=True)
    df = add_technical_indicators(df, tf)
    dfs[tf] = df

# Загружаем 15m
tf_15m = "15m"
filename_15m = f"BTCUSDT_{tf_15m}_3year.csv"
filepath_15m = os.path.join(RAW_DATA_DIR, filename_15m)
df_15m = pd.read_csv(filepath_15m, parse_dates=["open_time", "close_time"])
df_15m.set_index("open_time", inplace=True)
df_15m = add_technical_indicators(df_15m, tf_15m)

# Находим order blocks, FVG и уровни ликвидности на старших таймфреймах
order_blocks_1h = find_order_blocks(dfs["1h"])
fvg_4h = find_fvg(dfs["4h"])
liquidity_1d = find_liquidity_levels(dfs["1d"])

# Добавляем новые признаки в 15m
df_15m = add_order_block_feature(df_15m, order_blocks_1h)
df_15m = add_fvg_feature(df_15m, fvg_4h)
df_15m = add_liquidity_feature(df_15m, liquidity_1d)

# Добавляем технические признаки с других таймфреймов с суффиксами
for tf_higher in ["1h", "4h", "1d"]:
    df_h = dfs[tf_higher].add_suffix(f"_{tf_higher}")
    df_15m = df_15m.join(df_h, how="left")

df_15m.index.name = "open_time"

df_15m.to_csv(PROCESSED_DATA_PATH)
print(f"Processed features saved to {PROCESSED_DATA_PATH}")

