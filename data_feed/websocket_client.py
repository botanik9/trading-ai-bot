import asyncio
import websockets
import json
from typing import Callable, Optional


def parse_depth_update(data: dict) -> dict:
    """
    Преобразует сообщение depthUpdate в структурированный словарь.
    """
    return {
        'timestamp': data['E'],
        'symbol': data['s'],
        'bids': [(float(p), float(q)) for p, q in data['b'] if float(q) > 0],
        'asks': [(float(p), float(q)) for p, q in data['a'] if float(q) > 0],
    }


class BinanceWebSocketClient:
    """
    WebSocket клиент для получения стакана (depth) с Binance Futures.
    """
    def __init__(
        self,
        symbol: str = 'btcusdt',
        on_depth: Optional[Callable[[dict], None]] = None,
    ):
        self.symbol = symbol.lower()
        self.on_depth = on_depth
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth@100ms"

    async def _listen(self):
        print(f"🔌 Connecting to {self.ws_url}")
        try:
            async with websockets.connect(self.ws_url) as ws:
                print("✅ Connected to Binance WebSocket")
                async for message in ws:
                    data = json.loads(message)

                    if data.get('e') == 'depthUpdate' and self.on_depth:
                        parsed = parse_depth_update(data)
                        self.on_depth(parsed)

        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            await asyncio.sleep(5)
            await self._listen()  # Рекурсивный reconnect

    def run(self):
        """
        Запуск клиента (синхронный).
        """
        asyncio.get_event_loop().run_until_complete(self._listen())

