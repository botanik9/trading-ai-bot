import pandas as pd
import numpy as np

def add_technical_indicators(df, timeframe):
    df["close"] = pd.to_numeric(df["close"], errors='coerce')
    df["sma_14"] = df["close"].rolling(window=14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df

def find_order_blocks(df):
    """
    Находим order blocks — свечи с большими телами (более 75% квантиль),
    возвращаем список диапазонов (low, high).
    """
    ob = []
    body_size = (df["close"] - df["open"]).abs()
    threshold = body_size.quantile(0.75)  # большие свечи

    big_bodies = df[body_size >= threshold]
    for _, row in big_bodies.iterrows():
        low = row["low"]
        high = row["high"]
        ob.append((low, high))
    return ob

def find_fvg(df):
    """
    Ищем Fair Value Gaps — gaps между телами свечей.
    Возвращаем список mid точек gap.
    """
    fvg_zones = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        if curr["open"] > prev["close"]:
            gap_low = prev["close"]
            gap_high = curr["open"]
            mid = (gap_low + gap_high) / 2
            fvg_zones.append(mid)
        elif curr["close"] < prev["open"]:
            gap_low = curr["close"]
            gap_high = prev["open"]
            mid = (gap_low + gap_high) / 2
            fvg_zones.append(mid)
    return fvg_zones

def find_liquidity_levels(df):
    """
    Находим уровни ликвидности — локальные максимумы и минимумы.
    Возвращаем список уровней.
    """
    liquidity = []
    high = df["high"]
    low = df["low"]
    for i in range(1, len(df) - 1):
        if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
            liquidity.append(high.iloc[i])
        if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]:
            liquidity.append(low.iloc[i])
    return liquidity

def add_order_block_feature(df_15m, order_blocks):
    """
    Булева фича: 1 если close входит в любой order block, иначе 0
    """
    res = []
    for price in df_15m["close"]:
        inside = any(low <= price <= high for (low, high) in order_blocks)
        res.append(1 if inside else 0)
    df_15m["order_block"] = res
    return df_15m

def add_fvg_feature(df_15m, fvg_zones):
    """
    Расстояние от close до ближайшего FVG, если FVG нет — NaN
    """
    dist = []
    for price in df_15m["close"]:
        if not fvg_zones:
            dist.append(np.nan)
            continue
        min_dist = min(abs(price - mid) for mid in fvg_zones)
        dist.append(min_dist)
    df_15m["fvg_distance"] = dist
    return df_15m

def add_liquidity_feature(df_15m, liquidity_levels):
    """
    Расстояние от close до ближайшего уровня ликвидности, если уровней нет — NaN
    """
    dist = []
    for price in df_15m["close"]:
        if not liquidity_levels:
            dist.append(np.nan)
            continue
        min_dist = min(abs(price - lvl) for lvl in liquidity_levels)
        dist.append(min_dist)
    df_15m["liquidity_distance"] = dist
    return df_15m

