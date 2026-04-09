import pandas as pd
import numpy as np


def get_lagged_candles(df, lags=3):
    """
    Features dos últimos N candles: retorno e direção defasados.
    Captura padrões sequenciais simples como "3 candles vermelhos seguidos"
    ou "momentum de alta diminuindo".

    Inclui também:
    - gap: distância do Open atual ao Close anterior (pressão direcional de abertura)
    - consecutive_dir: conta quantos candles seguidos na mesma direção
    """
    pct_ret = df['Close'].pct_change()
    candle_dir = np.sign(df['Close'] - df['Open'])

    for lag in range(1, lags + 1):
        df[f'ret_lag_{lag}']  = pct_ret.shift(lag)
        df[f'dir_lag_{lag}']  = candle_dir.shift(lag)

    # Gap: Open atual vs Close anterior
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)

    # Contagem de candles consecutivos na mesma direção
    def count_consecutive(series):
        result = np.zeros(len(series))
        count = 0
        for i in range(len(series)):
            if i == 0:
                count = 1
            elif series.iloc[i] == series.iloc[i - 1] and series.iloc[i] != 0:
                count += 1
            else:
                count = 1
            result[i] = count * series.iloc[i]  # positivo=bull, negativo=bear
        return pd.Series(result, index=series.index)

    df['consecutive_dir'] = count_consecutive(candle_dir)

    return df
