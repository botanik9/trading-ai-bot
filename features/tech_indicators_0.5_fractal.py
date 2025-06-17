def find_fibonacci_retrace(df, lookback=20):
    """
    Находит импульсы (фракталы) и рассчитывает уровень 0.5 Фибоначчи.
    """
    fib_levels = []

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]

        high = window["high"].max()
        low = window["low"].min()
        last_close = df.iloc[i]["close"]

        fib_0_5 = low + (high - low) * 0.5

        # если цена была выше high и пошла вниз — фиксируем зону отката
        if last_close < high and last_close > fib_0_5:
            fib_levels.append({
                "index": i,
                "high": high,
                "low": low,
                "fib_0_5": fib_0_5,
                "price": last_close,
            })

    return fib_levels

