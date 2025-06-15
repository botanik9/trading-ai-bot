import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# Исправленная строка добавления пути
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '.')))

BASE_DIR = os.path.abspath(os.path.join(current_dir, '..'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(BASE_DIR, 'data/processed/processed_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/deep_resnet_model.pt')

# Параметры для сверхглубокой модели
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
SEQ_LEN = 60
DROPOUT_RATE = 0.4
HIDDEN_SIZE = 128
NUM_LAYERS = 20

# Улучшенная функция создания тензоров
def create_features_targets(df, sequence_length=SEQ_LEN):
    data = df.values
    X = np.array([data[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    y = np.array([data[i+sequence_length][0] for i in range(len(data) - sequence_length)])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc_layers(out)

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Загрузка и подготовка данных
    df = pd.read_csv(DATA_PATH)
    initial_count = len(df)
    df = df.dropna()
    print(f"Removed {initial_count - len(df)} rows with missing values")
    
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found")
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    X, y = create_features_targets(df[numeric_cols])
    
    dataset = TensorDataset(X, y)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)

    input_dim = X.shape[2]
    model = DeepLSTM(
        input_size=input_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=DROPOUT_RATE
    ).to(DEVICE)
    
    if DEVICE.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(DEVICE).total_memory
        allocated_mem = torch.cuda.memory_allocated()
        free_mem = total_mem - allocated_mem
        print(f"GPU Memory: Total: {total_mem/1e9:.2f}GB, Free: {free_mem/1e9:.2f}GB")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=3,
        factor=0.5
    )
    criterion = nn.HuberLoss()

    print(f"🚀 Training Deep LSTM with {NUM_LAYERS} layers")
    print(f"📊 Dataset: {len(dataset)} samples | Input dim: {input_dim}")
    print(f"⚙️ Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Saved best model with val loss: {val_loss:.6f}")

    print(f"✅ Training complete. Best model saved to {MODEL_PATH}")

if __name__ == '__main__':
    train()
