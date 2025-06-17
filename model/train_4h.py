import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import RobustScaler

# Пути
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '.')))
BASE_DIR = os.path.abspath(os.path.join(current_dir, '..'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(BASE_DIR, 'data/processed/processed_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/deep_resnet_model.pt')

# Параметры
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
SEQ_LEN = 256
DROPOUT_RATE = 0.4
HIDDEN_SIZE = 128
NUM_LAYERS = 32
NUM_HEADS = 8
CONV_CHANNELS = 64
L2_REG = 1e-5

# Создание признаков и целей (price и volume)
def create_features_targets(df, sequence_length=SEQ_LEN):
    data = df.values
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    # Предполагается, что первый столбец — price, второй — volume
    y_price = np.array([data[i+sequence_length][0] for i in range(len(data) - sequence_length)])
    y_volume = np.array([data[i+sequence_length][1] for i in range(len(data) - sequence_length)])
    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(y_price).float().unsqueeze(1),
        torch.from_numpy(y_volume).float().unsqueeze(1),
    )

class ResidualLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.input_proj(x)  # подгоняем размер входа, если нужно
        out, _ = self.lstm(x)
        out = out + residual  # residual connection
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.proj(out)
        return out

class DeepResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_heads, conv_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        current_input_size = input_size
        for _ in range(num_layers):
            layer = ResidualLSTMLayer(current_input_size, hidden_size, dropout)
            self.layers.append(layer)
            current_input_size = hidden_size
        
        # Multihead attention по времени
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        # Временные сверточные слои
        self.conv1 = nn.Conv1d(hidden_size, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Нелинейные проекции
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Две головы для price и volume
        self.fc_price = nn.Linear(hidden_size, 1)
        self.fc_volume = nn.Linear(hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        for layer in self.layers:
            x = layer(x)
        
        # MultiheadAttention expects (batch, seq, embed)
        attn_output, _ = self.attention(x, x, x)  # self-attention
        x = x + attn_output  # residual
        
        # Conv layers expect (batch, channels, seq)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu(self.conv1(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = self.relu(self.conv2(x_conv))
        x_conv = self.dropout(x_conv)
        x = x_conv.permute(0, 2, 1)
        
        x = self.proj(x)
        
        # Берём последний временной шаг
        x_last = x[:, -1, :]  # (batch, hidden_size)
        
        pred_price = self.fc_price(x_last)
        pred_volume = self.fc_volume(x_last)
        
        return pred_price, pred_volume

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    initial_count = len(df)
    df = df.dropna()
    print(f"Removed {initial_count - len(df)} rows with missing values")

    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        raise ValueError("Expect at least 2 numeric columns (price and volume)")

    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X, y_price, y_volume = create_features_targets(df[numeric_cols])

    dataset = TensorDataset(X, y_price, y_volume)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, num_workers=4, pin_memory=True)

    input_dim = X.shape[2]  # число фич (должно быть 48 или сколько у вас)
    model = DeepResidualLSTM(
        input_size=input_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        num_heads=NUM_HEADS,
        conv_channels=CONV_CHANNELS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')
    print(f"🚀 Training Deep Residual LSTM + Attention with {NUM_LAYERS} layers")
    print(f"📊 Dataset: {len(dataset)} samples | Input dim: {input_dim}")
    print(f"⚙️ Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_x, batch_price, batch_volume in train_loader:
            batch_x, batch_price, batch_volume = batch_x.to(DEVICE), batch_price.to(DEVICE), batch_volume.to(DEVICE)

            optimizer.zero_grad()
            pred_price, pred_volume = model(batch_x)
            loss_price = criterion(pred_price, batch_price)
            loss_volume = criterion(pred_volume, batch_volume)
            loss = loss_price + loss_volume
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_price, batch_volume in val_loader:
                batch_x, batch_price, batch_volume = batch_x.to(DEVICE), batch_price.to(DEVICE), batch_volume.to(DEVICE)
                pred_price, pred_volume = model(batch_x)
                loss_price = criterion(pred_price, batch_price)
                loss_volume = criterion(pred_volume, batch_volume)
                loss = loss_price + loss_volume
                val_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(epoch + val_loss)  # scheduler.step требует значение эпохи + метрики

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Saved best model with val loss: {val_loss:.6f}")

    print(f"✅ Training complete. Best model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train()
