import pandas as pd
import numpy as np


def get_momentum(df):
    """
    Retornos de preço em múltiplas janelas temporais.
    Captura a velocidade e direção do mercado em diferentes escalas.
    O XGBoost aprende automaticamente quais lags são mais preditivos.

    Janelas escolhidas para cobrir: intracandle, hora, meia-sessão,
    sessão, dia e dois dias (para dados 1H).
    """
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f'ret_{lag}'] = df['Close'].pct_change(lag)

    # Aceleração do momentum (diferença entre momentum curto e longo)
    df['mom_accel_6_24'] = df['ret_6'] - df['ret_24']   # curto - longo
    df['mom_accel_3_12'] = df['ret_3'] - df['ret_12']

    return df
