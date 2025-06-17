import pandas as pd
import numpy as np

def detect_market_condition(df):
    """
    Определение тренда или боковика на основе SMA и ATR.
    """
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["atr_14"] = (df["high"] - df["low"]).rolling(window=14).mean()
    atr_threshold = df["atr_14"].mean()

    df["market_condition"] = np.where(
        (df["close"] > df["sma_20"]) & (df["atr_14"] > atr_threshold),
        "trend",
        "range"
    )
    return df

def find_fvg_ifvg(df, impulse_threshold=0.01):
    """
    Поиск FVG и IFVG зон (IFVG = импульсные FVG).
    """
    fvg = []
    ifvg = []
    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        change = abs(curr["close"] - prev["close"]) / prev["close"]

        if curr["open"] > prev["close"]:
            mid = (prev["close"] + curr["open"]) / 2
            (ifvg if change > impulse_threshold else fvg).append(mid)

        elif curr["close"] < prev["open"]:
            mid = (curr["close"] + prev["open"]) / 2
            (ifvg if change > impulse_threshold else fvg).append(mid)
    
    return ifvg, fvg

def find_equal_highs_lows(df, tolerance=0.001):
    """
    Поиск равных максимумов и минимумов.
    """
    eq_highs = []
    eq_lows = []
    highs = df["high"].round(5)
    lows = df["low"].round(5)

    for i in range(2, len(df)):
        h1, h2 = highs.iloc[i - 1], highs.iloc[i]
        l1, l2 = lows.iloc[i - 1], lows.iloc[i]
        
        if abs(h1 - h2) / h1 < tolerance:
            eq_highs.append(h1)
        
        if abs(l1 - l2) / l1 < tolerance:
            eq_lows.append(l1)

    return list(set(eq_highs)), list(set(eq_lows))

def filter_broken_levels(df, eq_highs, eq_lows):
    """
    Удаление пробитых уровней equal highs/lows.
    """
    valid_highs = []
    valid_lows = []
    
    for level in eq_highs:
        if all(df["high"] < level * 1.0005):  # не пробит
            valid_highs.append(level)

    for level in eq_lows:
        if all(df["low"] > level * 0.9995):  # не пробит
            valid_lows.append(level)

    return valid_highs, valid_lows

def build_4h_indicator(df_4h):
    """
    Основной индикатор: тренд/флэт, IFVG, FVG, неповрежденные EQ уровни.
    """
    df = df_4h.copy()
    df = detect_market_condition(df)
    ifvg, fvg = find_fvg_ifvg(df)
    eq_highs, eq_lows = find_equal_highs_lows(df)
    eq_highs, eq_lows = filter_broken_levels(df, eq_highs, eq_lows)

    indicator = {
        "market_condition": df["market_condition"].iloc[-1],  # 'trend' или 'range'
        "ifvg": ifvg,
        "fvg": fvg,
        "equal_highs": eq_highs,
        "equal_lows": eq_lows
    }
    return indicator
