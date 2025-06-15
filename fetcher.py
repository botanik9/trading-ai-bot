import asyncio
import json
import websockets
from collections import namedtuple

Candle = namedtuple('Candle', ['open', 'high', 'low', 'close', 'volume'])

async def candle_stream(symbol="btcusdt", interval="1m"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            kline = data.get("k")
            if kline and kline.get("x"):  # проверяем, что свеча закрыта
                candle = Candle(
                    open=float(kline['o']),
                    high=float(kline['h']),
                    low=float(kline['l']),
                    close=float(kline['c']),
                    volume=float(kline['v'])
                )
                yield candle

# Пример использования (если запускать отдельно)
# async def main():
#     async for candle in candle_stream():
#         print(candle)

# asyncio.run(main())
