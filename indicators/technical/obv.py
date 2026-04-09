import pandas as pd
import numpy as np

def get_obv(df):
    # Usamos rolling sum ao invés de cumsum para evitar data leakage.
    # cumsum() no dataset inteiro antes do split treino/teste faz com que
    # o OBV de qualquer ponto dependa de dados futuros (do conjunto de teste).
    signed_vol = np.sign(df['Close'].diff()) * df['Volume']
    df['OBV_Slope'] = signed_vol.rolling(5).sum()

    return df