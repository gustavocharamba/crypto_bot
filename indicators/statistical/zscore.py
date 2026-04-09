import pandas as pd
import numpy as np

def get_zscore(df, window=30):
    # ZScore sobre retornos log (estacionário) em vez de preço absoluto.
    # Preço absoluto (Close) é não-estacionário: o modelo aprende padrões
    # ligados ao nível de preço, que proxy para o período histórico.
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    z_mean = log_ret.rolling(window=window).mean()
    z_std  = log_ret.rolling(window=window).std()

    df['ZScore'] = (log_ret - z_mean) / z_std.replace(0, np.nan)

    return df