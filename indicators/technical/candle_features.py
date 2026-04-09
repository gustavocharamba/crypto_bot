import pandas as pd
import numpy as np


def get_candle_features(df):
    """
    Features de price action baseadas na estrutura de cada candle.
    Capturam a 'intenção' do mercado sem qualquer look-ahead.

    - body_ratio:  tamanho do corpo relativo ao range total (0=doji, 1=marubozu)
    - upper_wick:  sombra superior relativa ao range (rejeição de alta)
    - lower_wick:  sombra inferior relativa ao range (suporte/rejeição de baixa)
    - body_dir:    direção do candle (+1 bull, -1 bear)
    """
    high_low = df['High'] - df['Low'] + 1e-9  # evita divisão por zero

    body_top    = df[['Open', 'Close']].max(axis=1)
    body_bottom = df[['Open', 'Close']].min(axis=1)

    df['candle_body_ratio'] = (df['Close'] - df['Open']).abs() / high_low
    df['candle_upper_wick'] = (df['High'] - body_top)    / high_low
    df['candle_lower_wick'] = (body_bottom - df['Low'])  / high_low
    df['candle_dir']        = np.sign(df['Close'] - df['Open'])

    return df
