import torch
from model.architecture import SimpleLSTM
from buffer import CandleBuffer
import asyncio
from fetcher import candle_stream

TIMEFRAME_LENGTH = 12

def preprocess_features(features):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return x

async def main():
    model = SimpleLSTM(input_size=1, hidden_size=64, output_size=1)
    model.load_state_dict(torch.load('model/checkpoints/model.pt'))
    model.eval()

    buffer = CandleBuffer(maxlen=TIMEFRAME_LENGTH)

    async for candle in candle_stream():
        buffer.add(candle)

        if not buffer.is_full():
            print("Ждем заполнения буфера...")
            continue

        features = buffer.get_features()
        X = preprocess_features(features)

        with torch.no_grad():
            pred = model(X)

        print(f"Prediction: {pred.numpy()}")

if __name__ == '__main__':
    asyncio.run(main())

