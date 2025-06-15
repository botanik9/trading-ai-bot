import sys
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from architecture import SimpleLSTM

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(BASE_DIR, 'data/processed/processed_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/model.pt')
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQ_LEN = 50

def create_features_targets(df, sequence_length=SEQ_LEN):
    data = df.values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length][0])  # например, close_15m
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл с данными не найден: {DATA_PATH}. Пожалуйста, подготовьте данные в папке data/processed."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # Если есть timestamp, удалим/сделаем индекс, но он не нужен в обучении
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    # Оставляем только числовые столбцы
    df = df.select_dtypes(include=['number'])
    if df.empty:
        raise ValueError("После фильтрации остались пустые данные. Проверьте содержимое CSV.")

    X, y = create_features_targets(df)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleLSTM(input_size=X.shape[2], hidden_size=64, output_size=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Модель сохранена в {MODEL_PATH}")

if __name__ == '__main__':
    train()

