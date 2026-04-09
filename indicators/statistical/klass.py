import pandas as pd
import numpy as np

def get_klass(df, window=20):

    df = df.copy()

    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    rs = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    df[f'GKVol_{window}'] = rs.rolling(window).mean() ** 0.5

    return df
