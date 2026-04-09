import pandas as pd
import numpy as np


def get_autocorr(df, window=24):
    """
    Autocorrelação serial dos log-retornos em lag 1.
    Detecta se o mercado está em regime de momentum (+) ou reversão (-).

    - autocorr > 0: retornos positivos tendem a seguir retornos positivos (trending)
    - autocorr < 0: retornos negativo tendem a seguir retornos positivos (mean-reverting)
    - autocorr ≈ 0: mercado sem memória serial (eficiente no curto prazo)

    Essa feature permite ao modelo condicionar sua predição ao regime atual.
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    df[f'ret_autocorr_{window}'] = log_ret.rolling(window).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 4 else np.nan,
        raw=False
    )

    return df
