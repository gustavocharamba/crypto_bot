import pandas as pd
import numpy as np


def get_range_expansion(df, atr_col='ATR_14'):
    """
    Range Expansion Ratio: range do candle atual relativo ao ATR.
    Detecta breakouts (ratio alto) e consolidações (ratio baixo).

    ratio > 1.5 → candle de expansão/breakout
    ratio < 0.5 → candle de contração/consolidação
    """
    if atr_col not in df.columns:
        # Fallback: calcula ATR 14 se não existir
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift(1)).abs()
        low_close  = (df['Low']  - df['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
    else:
        atr = df[atr_col]

    candle_range = df['High'] - df['Low']
    df['range_expansion'] = candle_range / (atr + 1e-9)

    # Versão suavizada (média de curto prazo)
    df['range_expansion_3'] = df['range_expansion'].rolling(3).mean()

    return df
