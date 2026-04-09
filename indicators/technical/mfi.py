import pandas as pd
import numpy as np


def get_mfi(df, window=14):
    """
    Money Flow Index (MFI) — RSI ponderado por volume.
    Diferencia pressão compradora de vendedora com base no volume real,
    sendo mais informativo que RSI puro em mercados como BTC onde
    volume é altamente relevante.

    MFI > 80: sobrecomprado (possível reversão de alta)
    MFI < 20: sobrevendido (possível reversão de baixa)
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']

    # Fluxo positivo/negativo baseado na direção do preço típico
    tp_change = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_change > 0, 0)
    neg_flow = raw_money_flow.where(tp_change < 0, 0)

    pos_mf = pos_flow.rolling(window).sum()
    neg_mf = neg_flow.rolling(window).sum()

    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))
    df[f'MFI_{window}'] = mfi

    # Normalizado entre -1 e 1 (mais conveniente para o modelo)
    df[f'MFI_{window}_norm'] = (mfi - 50) / 50

    return df
