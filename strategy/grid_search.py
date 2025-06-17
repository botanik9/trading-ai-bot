# File: strategy/grid_search.py

import os
import subprocess
import pandas as pd
from itertools import product

# --- Диапазоны для перебора ---
PARAM_GRID = {
    "ENTRY_THRESHOLD": [0.003],
    "TAKE_PROFIT":     [0.003],
    "STOP_LOSS":       [0.005],
    "MIN_HOLD_BARS":   [2],
    "MAX_HOLD_BARS":   [12]
}

# --- Подготовка ---
RESULTS_DIR = "/home/trading/trading-ai-bot/results"  # ← Абсолютный путь
os.makedirs(RESULTS_DIR, exist_ok=True)              # ← Создаём, если нет

param_combinations = list(product(*PARAM_GRID.values()))
total_runs = len(param_combinations)

print(f"\n🔍 GridSearch начат")
print(f"🧪 Всего комбинаций: {total_runs}\n")

# --- DataFrame для результатов ---
results_df = pd.DataFrame(columns=list(PARAM_GRID.keys()) + [
    "Final_Balance", "Profit_Percent", "Win_Rate",
    "Profit_Factor", "Max_Drawdown", "Total_Trades"
])

# --- Цикл прогонов ---
for idx, params in enumerate(param_combinations):
    print(f"\n🔁 Прогон {idx+1}/{total_runs}")
    
    tmp_output = os.path.join(RESULTS_DIR, f"tmp_{idx}.csv")  # ← Точно в нужную папку

    cmd = [
        "python", "backtest.py",
        "--ENTRY_THRESHOLD", str(params[0]),
        "--TAKE_PROFIT", str(params[1]),
        "--STOP_LOSS", str(params[2]),
        "--MIN_HOLD_BARS", str(params[3]),
        "--MAX_HOLD_BARS", str(params[4]),
        "--output", tmp_output
    ]

    subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if os.path.exists(tmp_output):
        result = pd.read_csv(tmp_output)
        for key, val in zip(PARAM_GRID.keys(), params):
            result[key] = val
        results_df = pd.concat([results_df, result], ignore_index=True)
        os.remove(tmp_output)

# --- Сохраняем финальный CSV ---
results_df.to_csv(os.path.join(RESULTS_DIR, "gridsearch_results.csv"), index=False)
print("\n✅ GridSearch завершён!")
print("📊 Результаты сохранены: /home/trading/trading-ai-bot/results/gridsearch_results.csv")
