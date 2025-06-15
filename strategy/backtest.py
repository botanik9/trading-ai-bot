import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.train import DeepLSTM, DEVICE, SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'deep_resnet_model.pt')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_features.csv')

INITIAL_BALANCE = 1000
TRADE_SIZE = 1.0
THRESHOLD = 0.003  # чувствительность к сигналу

def create_sequences(data, sequence_length=SEQ_LEN):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    return np.array(X)

def calculate_max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd * 100  # в %

def backtest():
    df = pd.read_csv(DATA_PATH)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df = df[numeric_cols].dropna()
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    data = df.values
    X_all = create_sequences(data)
    X_all = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    model = DeepLSTM(
        input_size=X_all.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=DROPOUT_RATE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(X_all).squeeze().cpu().numpy()

    balance = INITIAL_BALANCE
    equity_curve = [balance]
    position = None
    entry_price = 0
    win_trades = 0
    loss_trades = 0

    prices = df.iloc[SEQ_LEN:, 0].values  # предполагаем, что первый столбец — цена

    for i in range(len(preds) - 1):
        pred_diff = preds[i+1] - preds[i]
        price = prices[i]

        if position is None:
            if pred_diff > THRESHOLD:
                position = 'long'
                entry_price = price
            elif pred_diff < -THRESHOLD:
                position = 'short'
                entry_price = price
        else:
            if position == 'long' and pred_diff < -THRESHOLD:
                pnl = (price - entry_price) * TRADE_SIZE
                balance += pnl
                if pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                position = None
            elif position == 'short' and pred_diff > THRESHOLD:
                pnl = (entry_price - price) * TRADE_SIZE
                balance += pnl
                if pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                position = None
        
        equity_curve.append(balance)

    print(f"📈 Final Balance: ${balance:.2f}")
    print(f"✅ Total Trades: {win_trades + loss_trades}")
    print(f"✔️ Win Trades: {win_trades}, Loss Trades: {loss_trades}")
    print(f"📉 Max Drawdown: {calculate_max_drawdown(equity_curve):.2f}%")

    # Сохраняем график
    os.makedirs("plots", exist_ok=True)
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/equity_curve.png")
    print("🖼️ Equity chart saved to: plots/equity_curve.png")

if __name__ == "__main__":
    backtest()

