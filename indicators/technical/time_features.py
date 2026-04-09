import pandas as pd
import numpy as np


def get_time_features(df):
    """
    Features cíclicas de tempo para capturar sazonalidade intradiária e semanal.
    Usa sin/cos para representar ciclos de forma contínua (hora 23 ≈ hora 0).

    BTC opera 24/7, mas tem padrões claros por sessão:
    - Ásia:   00h-08h UTC  (volume/volatilidade menores)
    - Europa: 08h-16h UTC  (volume crescente)
    - EUA:    13h-21h UTC  (maior volatilidade, sobreposição)
    """
    idx = df.index

    # Hora do dia (0-23) → ciclo de 24h
    hour = idx.hour
    df['time_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['time_hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Dia da semana (0=seg, 6=dom) → ciclo de 7 dias
    dow = idx.dayofweek
    df['time_dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['time_dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return df
