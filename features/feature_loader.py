import json
import os
import pandas as pd

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "feature_registry.json")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/processed_features.csv")

def load_feature_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    return registry

def load_processed_features():
    df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    return df

def get_features_for_timeframes(df, timeframes):
    registry = load_feature_registry()
    features = []
    for tf in timeframes:
        features += registry["features"].get(tf, [])

    # Отфильтровываем строки без NaN по признакам
    X = df[features].dropna()

    # Предполагается, что целевая переменная называется 'target'
    if 'target' not in df.columns:
        raise ValueError("Колонка 'target' не найдена в датафрейме.")

    # Подгоняем y по тем же индексам, что и X
    y = df.loc[X.index, 'target']

    return X, y

# Тест при запуске напрямую
if __name__ == "__main__":
    timeframes = ["1d", "4h", "1h", "15m"]
    df = load_processed_features()
    X, y = get_features_for_timeframes(df, timeframes)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.head())
    print(y.head())

