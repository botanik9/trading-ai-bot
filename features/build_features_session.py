import pandas as pd
import os

# Импорт технических индикаторов и сессий
from tech_indicators_session import (
    add_technical_indicators,
    add_trading_sessions
)

# Установим базовые пути
BASE_DIR = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/processed_features.csv")

# Загружаем и обрабатываем данные для 4h
tf = "4h"
filename = f"BTCUSDT_{tf}_3year.csv"
filepath = os.path.join(RAW_DATA_DIR, filename)
df = pd.read_csv(filepath, parse_dates=["open_time", "close_time"])

# Устанавливаем индекс времени
df.set_index("open_time", inplace=True)
df.index = df.index.tz_localize("UTC")  # обязательно в UTC

# Добавляем технические индикаторы
df = add_technical_indicators(df, tf)

# Добавляем торговые сессии
df = add_trading_sessions(df)

# Остальные фичи — временно отключены, так как функций пока нет
# from other_features import (
#     find_order_blocks,
#     find_fvg,
#     find_liquidity_levels,
#     add_order_block_feature,
#     add_fvg_feature,
#     add_liquidity_feature
# )

# order_blocks_4h = find_order_blocks(df)
# fvg_4h = find_fvg(df)
# liquidity_1d = find_liquidity_levels(df)

# df = add_order_block_feature(df, order_blocks_4h)
# df = add_fvg_feature(df, fvg_4h)
# df = add_liquidity_feature(df, liquidity_1d)

# Сохраняем результат
df.index.name = "open_time"
df.to_csv(PROCESSED_DATA_PATH)
print(f"Processed features for 4h saved to {PROCESSED_DATA_PATH}")
