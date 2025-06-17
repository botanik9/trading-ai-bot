import pandas as pd

def add_trading_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None or str(df.index.tz) != 'UTC':
        df.index = df.index.tz_localize('UTC')  # Убедиться, что индекс — UTC

    # Добавим колонку сессии на основе UTC-времени
    df['session'] = df.index.hour.map(assign_session)
    return df

def assign_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "New York"
    else:
        return "CME"

def add_technical_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Простейшая реализация тех. индикаторов (SMA и EMA) на закрытии.
    """
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    return df
