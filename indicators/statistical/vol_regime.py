import pandas as pd
import numpy as np


def get_vol_regime(df, short_window=12, long_window=72):
    """
    Volatility Regime: razão entre volatilidade de curto e longo prazo.
    Detecta se o mercado está em regime de alta ou baixa volatilidade.

    ratio > 1.2 → mercado "elétrico", expansão de volatilidade
    ratio < 0.8 → mercado calmo, contração de volatilidade
    ratio ≈ 1.0 → volatilidade normal

    O modelo aprende estratégias diferentes para cada regime.
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    vol_short = log_ret.rolling(short_window).std()
    vol_long  = log_ret.rolling(long_window).std()

    df['vol_regime'] = vol_short / (vol_long + 1e-9)

    # Log da razão (mais simétrico e estacionário)
    df['vol_regime_log'] = np.log(df['vol_regime'].clip(lower=1e-4))

    return df
