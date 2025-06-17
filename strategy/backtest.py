# File: strategy/backtest.py

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

# Добавляем путь к модели
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model.train import DeepResidualLSTM, DEVICE, SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE

# --- Параметры стратегии ---
INITIAL_BALANCE = 1000
TRADE_SIZE = 100.0
ENTRY_THRESHOLD = 0.005
EXIT_THRESHOLD = 0.003
TAKE_PROFIT = 0.010
STOP_LOSS = 0.006
MIN_HOLD_BARS = 3
MAX_HOLD_BARS = 24
CHUNK_SIZE = 256
RISK_MANAGEMENT = False
COMMISSION = 0.0005  # 0.05%

# --- Парсер аргументов ---
def parse_args():
    parser = argparse.ArgumentParser(description="Backtest с параметрами")
    parser.add_argument("--ENTRY_THRESHOLD", type=float, default=ENTRY_THRESHOLD)
    parser.add_argument("--EXIT_THRESHOLD", type=float, default=EXIT_THRESHOLD)
    parser.add_argument("--TAKE_PROFIT", type=float, default=TAKE_PROFIT)
    parser.add_argument("--STOP_LOSS", type=float, default=STOP_LOSS)
    parser.add_argument("--MIN_HOLD_BARS", type=int, default=MIN_HOLD_BARS)
    parser.add_argument("--MAX_HOLD_BARS", type=int, default=MAX_HOLD_BARS)
    parser.add_argument("--output", type=str, help="Файл для сохранения результата", default=None)
    return parser.parse_args()

# --- Создание последовательностей ---
def create_sequences(data, sequence_length=SEQ_LEN):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    return np.array(X)

# --- Расчёт Drawdown ---
def calculate_max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (value - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd * 100

# --- Основная функция ---
def backtest(args):
    ENTRY_THRESHOLD = args.ENTRY_THRESHOLD
    EXIT_THRESHOLD = args.EXIT_THRESHOLD
    TAKE_PROFIT = args.TAKE_PROFIT
    STOP_LOSS = args.STOP_LOSS
    MIN_HOLD_BARS = args.MIN_HOLD_BARS
    MAX_HOLD_BARS = args.MAX_HOLD_BARS

    print("⏳ Loading data...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'processed_features.csv'))
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'])
        df = df.drop(columns=['timestamp'])

    numeric_cols = df.select_dtypes(include=np.number).columns
    df = df[numeric_cols].dropna()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    data = df.values
    X_all = create_sequences(data)
    X_all = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    print("🧠 Loading model...")
    model = DeepResidualLSTM(
        input_size=X_all.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        num_heads=4,
        conv_channels=64
    ).to(DEVICE)

    model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'checkpoints', 'deep_resnet_model.pt'), map_location=DEVICE))
    model.eval()

    print("🔮 Making predictions...")
    pred_price = np.zeros(len(X_all))

    for i in tqdm(range(0, len(X_all), CHUNK_SIZE), desc="Predicting"):
        chunk = X_all[i:i+CHUNK_SIZE]
        with torch.no_grad():
            chunk_pred = model(chunk)[0]
            pred_price[i:i+len(chunk_pred)] = chunk_pred.cpu().numpy().flatten()

    print("📊 Running backtest...")
    balance = INITIAL_BALANCE
    equity_curve = [balance]
    position = None
    entry_price = 0
    win_trades = loss_trades = hold_bars = 0
    prices = df.iloc[SEQ_LEN:].copy()
    prices['pred_diff'] = np.append(0, np.diff(pred_price))
    trades = []

    for i, row in prices.iterrows():
        current_price = row['close']
        pred_diff = row['pred_diff']

        trade_size = TRADE_SIZE
        if RISK_MANAGEMENT:
            trade_size = min(TRADE_SIZE, (balance * 0.02) / (current_price * STOP_LOSS))

        if position is None:
            if pred_diff > ENTRY_THRESHOLD:
                position = 'long'
                entry_price = current_price
                hold_bars = 0
                balance -= trade_size * current_price * COMMISSION
            elif pred_diff < -ENTRY_THRESHOLD:
                position = 'short'
                entry_price = current_price
                hold_bars = 0
                balance -= trade_size * current_price * COMMISSION

        else:
            hold_bars += 1
            pnl_pct = (current_price - entry_price)/entry_price if position == 'long' \
                      else (entry_price - current_price)/entry_price

            exit_condition = (
                (pred_diff < -EXIT_THRESHOLD if position == 'long' else pred_diff > EXIT_THRESHOLD) or
                pnl_pct >= TAKE_PROFIT or
                -pnl_pct >= STOP_LOSS or
                hold_bars >= MAX_HOLD_BARS
            )

            if exit_condition and hold_bars >= MIN_HOLD_BARS:
                pnl = pnl_pct * trade_size
                balance += pnl - trade_size * current_price * COMMISSION

                trades.append({
                    'type': position,
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'bars': hold_bars
                })

                if pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                position = None

        equity_curve.append(balance)

    print("\n💹 Backtest Results:")
    print(f"💰 Final Balance: ${balance:.2f}")
    total_trades = win_trades + loss_trades
    print(f"📈 Profit: {(balance / INITIAL_BALANCE - 1) * 100:.2f}%")
    print(f"📊 Total Trades: {total_trades}")
    print(f"✅ Win Rate: {win_trades / total_trades * 100:.1f}%")

    profit_factor = 0
    if trades:
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0])
        avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] < 0])
        profit_factor = avg_win * win_trades / (avg_loss * loss_trades) if avg_loss != 0 else float('inf')
        print(f"📊 Profit Factor: {profit_factor:.2f}")

    print(f"📉 Max Drawdown: {calculate_max_drawdown(equity_curve):.2f}%")

    # Сохранение результата
    metrics = {
        "ENTRY_THRESHOLD": ENTRY_THRESHOLD,
        "EXIT_THRESHOLD": EXIT_THRESHOLD,
        "TAKE_PROFIT": TAKE_PROFIT,
        "STOP_LOSS": STOP_LOSS,
        "MIN_HOLD_BARS": MIN_HOLD_BARS,
        "MAX_HOLD_BARS": MAX_HOLD_BARS,
        "Final_Balance": balance,
        "Profit_Percent": (balance / INITIAL_BALANCE - 1) * 100,
        "Win_Rate": win_trades / total_trades * 100 if total_trades > 0 else 0,
        "Profit_Factor": profit_factor,
        "Max_Drawdown": calculate_max_drawdown(equity_curve),
        "Total_Trades": total_trades
    }

    # Безопасное сохранение в CSV
    if args.output:
        try:
            df_result = pd.DataFrame([metrics])
            df_result.to_csv(args.output, index=False)
        except OSError:
            print("⚠️ Не удалось сохранить результат — ошибка доступа к директории")
    else:
        print("ℹ️ Результат не сохранён — файл вывода не указан")

if __name__ == "__main__":
    args = parse_args()
    backtest(args)
