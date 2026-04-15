import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from processing.pre_processing import get_preprocessing
from indicators.technical.loader_technical import get_technical_indicators
from indicators.statistical.loader_statistical import get_statistical_indicators
from indicators.market.loader_market import get_market_indicators
from processing.target import get_target
from validation.walk_forward import walk_forward
from validation.evaluate_thresholds import evaluate_thresholds
from validation.feature_importance import get_feature_importance
from models.xgboost_model import xgboost_model


def run_pipeline_xgb(data_path, horizon, name, timeframe="1h"):
    """Executa o pipeline e retorna (preds_series, y_real)."""
    df = pd.read_csv(data_path)
    df = get_preprocessing(df)
    df = get_technical_indicators(df)
    df = get_statistical_indicators(df)
    df = get_market_indicators(df, data_dir, timeframe=timeframe)
    df = get_target(df, horizon, name)
    df = df.dropna()

    y = df['Target']
    y_real = df['RealizedReturn']
    X = df.drop(columns=['Target', 'Return', 'RealizedReturn', 'Open', 'High', 'Low',
                          'Close', 'Adj Close', 'Volume', 'Date'], errors='ignore')

    preds, avg_metrics, last_model = walk_forward(
        xgboost_model, X, y, train_size=0.7, step_size=0.1, purge_window=horizon
    )

    print(f"\n===== XGBoost Classification Performance: {name} =====")
    print(avg_metrics)
    print("=======================================")

    if last_model is not None:
        try:
            fi = get_feature_importance(last_model, X, top_n=15)
            print(f"\n--- Top 15 Features ({name}) ---")
            print(fi.to_string(index=False))
        except Exception as e:
            print(f"  [Feature importance não disponível: {e}]")

    # Retorna Series com índice datetime para alinhamento multi-TF
    preds_series = pd.Series(preds, index=y.index, name=name)

    return preds_series, y_real


# ── Execução ─────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir  = os.path.join(base_dir, "data")

preds_1h, y_real_1h = run_pipeline_xgb(os.path.join(data_dir, "btcusd_1h.csv"),  12, "1H", timeframe="1h")
preds_4h, y_real_4h = run_pipeline_xgb(os.path.join(data_dir, "btcusd_4h.csv"),  18, "4H", timeframe="4h")
preds_1d, y_real_1d = run_pipeline_xgb(os.path.join(data_dir, "btcusd_1d.csv"),   7, "1D", timeframe="1d")

# ── Alinhamento multi-timeframe ───────────────────────────────────────────────
# Para cada candle 1H, associar a predição 4H mais recente (forward fill)
# e a predição 1D mais recente.
preds_4h_aligned = preds_4h.resample('1h').last().reindex(preds_1h.index, method='ffill')
preds_1d_aligned = preds_1d.resample('1h').last().reindex(preds_1h.index, method='ffill')

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