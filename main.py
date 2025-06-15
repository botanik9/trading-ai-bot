from data_feed.websocket_client import BinanceWebSocketClient
from pprint import pprint

def handle_depth(data):
    print("📥 Depth update:")
    pprint(data)

if __name__ == "__main__":
    client = BinanceWebSocketClient(symbol='BTCUSDT', on_depth=handle_depth)
    client.run()

