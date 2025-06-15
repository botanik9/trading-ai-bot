import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from architecture import SimpleLSTM
from features.feature_loader import load_processed_features, get_features_for_timeframes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMEFRAMES = ['1m', '5m']  # пример

def predict():
    df = load_processed_features()
    X, _ = get_features_for_timeframes(df, TIMEFRAMES)  # y не нужен

    X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
    X = X.unsqueeze(1)

    model = SimpleLSTM(input_size=X.shape[2], hidden_size=64, output_size=1).to(DEVICE)
    model.load_state_dict(torch.load("model/checkpoints/model.pt", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        prediction = model(X)
        print("Prediction:", prediction.cpu().numpy())

if __name__ == "__main__":
    predict()

