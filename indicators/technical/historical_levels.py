import pandas as pd
import numpy as np


def get_historical_levels(df, window_long=2190, window_short=168):
    """
    Distância do preço atual em relação a níveis históricos relevantes.
    Captura contexto de onde o preço está no seu range histórico.

    Parâmetros (ajustados para 1H por padrão):
    - window_long:  2190 candles ≈ 3 meses de 1H  (níveis macro)
    - window_short:  168 candles ≈ 1 semana de 1H  (níveis de curto prazo)

    Para 4H: use window_long=540 (90 dias), window_short=42 (1 semana)
    Para 1D: use window_long=90 (3 meses),  window_short=7  (1 semana)
    """
    # Máxima e mínima de longo prazo
    roll_high_l = df['Close'].rolling(window_long).max()
    roll_low_l  = df['Close'].rolling(window_long).min()

    df[f'dist_high_{window_long}'] = (df['Close'] / roll_high_l) - 1  # 0 = tá na máxima
    df[f'dist_low_{window_long}']  = (df['Close'] / roll_low_l)  - 1  # 0 = tá na mínima

    # Máxima e mínima de curto prazo  
    roll_high_s = df['Close'].rolling(window_short).max()
    roll_low_s  = df['Close'].rolling(window_short).min()

    df[f'dist_high_{window_short}'] = (df['Close'] / roll_high_s) - 1
    df[f'dist_low_{window_short}']  = (df['Close'] / roll_low_s)  - 1

    # Posição relativa dentro do range de curto prazo (0=mínima, 1=máxima)
    range_s = roll_high_s - roll_low_s + 1e-9
    df[f'price_position_{window_short}'] = (df['Close'] - roll_low_s) / range_s

    return df
