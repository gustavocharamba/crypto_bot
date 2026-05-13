import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from processing.pre_processing import get_preprocessing
from indicators.loader_technical import get_technical_indicators
from indicators.loader_statistical import get_statistical_indicators
from processing.get_target import get_target
from models.xgboost_model import get_xgboost_model


def run_pipeline_xgb(data_path, horizon, name, timeframe):
    df = pd.read_csv(data_path)
    df = get_preprocessing(df)
    df = get_technical_indicators(df)
    df = get_statistical_indicators(df)
    df = get_target(df, horizon, name)
    df = df.dropna()

    y = df['Target']
    y_real = df['RealizedReturn']
    X = df.drop(columns=['Target', 'Raw_Target', 'Return', 'RealizedReturn', 'Open', 'High', 'Low',
                          'Close', 'Adj Close', 'Volume', 'Date'], errors='ignore')

data_dir = "../data"

preds_30m, y_real_30m - run_pipeline_xgb(os.path.join(data_dir, "btc_30m.csv"), 240, "30M", "30m")
preds_1h, y_real_1h = run_pipeline_xgb(os.path.join(data_dir, "btc_1d.csv"),  120, "1H", "1h")
preds_2h, y_real_2h - run_pipeline_xgb(os.path.join(data_dir, "btc_2h.csv"), 60, "2H", "2h")
preds_4h_real_4h - run_pipeline_xgb(os.path.join(data_dir, "btc_4h.csv"), 30, "4h", "4H")
preds_1D, y_real_1D - run_pipeline_xgb(os.path.join(data_dir, "btc_30m.csv"), 5, "1D", "1d")

# ── Alinhamento multi-timeframe ───────────────────────────────────────────────
preds_1h_aligned = preds_1h.resample('30m').last().reindex(preds_1h.index, method='ffill')
preds_2h_aligned = preds_2h.resample('30m').last().reindex(preds_2h.index, method='ffill')
preds_4h_aligned = preds_4h.resample('30m').last().reindex(preds_4h.index, method='ffill')
preds_1d_aligned = preds_1d.resample('30m').last().reindex(preds_1d.index, method='ffill')


# Filtrar apenas onde há predições válidas em todos os TFs
valid_1h = ~np.isnan(preds_1h)
preds_1h_clean   = preds_1h[valid_1h].values
y_real_1h_clean  = y_real_1h[valid_1h].values
preds_4h_clean   = preds_4h_aligned[valid_1h].values
preds_1d_clean   = preds_1d_aligned[valid_1h].values

# ── Backtest multi-TF (1H sinal + 4H confirmação) ────────────────────────────
print("\n===== Melhores Thresholds 1H (standalone) =====")
results_1h_solo = evaluate_thresholds(preds_1h_clean, y_real_1h_clean, fee=0.0004)
print(results_1h_solo.head(10))

print("\n===== Melhores Thresholds 1H + 4H confirmação =====")
results_1h_4h = evaluate_thresholds(
    preds_1h_clean, y_real_1h_clean,
    preds_4h=preds_4h_clean,
    fee=0.0004
)
print(results_1h_4h.head(10))

print("\n===== Melhores Thresholds 1H + 4H + 1D confirmação =====")
results_1h_4h_1d = evaluate_thresholds(
    preds_1h_clean, y_real_1h_clean,
    preds_4h=preds_4h_clean,
    preds_1d=preds_1d_clean,
    fee=0.0004
)
print(results_1h_4h_1d.head(10))

print("\n===== Melhores Thresholds 4H (standalone) =====")
valid_4h = ~np.isnan(preds_4h)
results_4h = evaluate_thresholds(preds_4h[valid_4h].values, y_real_4h[valid_4h].values, fee=0.0004)
print(results_4h.head(10))