import pandas as pd

def add_technical_indicators(df, timeframe):
    # Пример индикаторов: SMA(14), RSI(14)
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
