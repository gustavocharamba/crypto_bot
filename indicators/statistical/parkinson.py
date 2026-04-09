import pandas as pd
import numpy as np

def get_parkinson(df, window=20):

    df = df.copy()
    hl_log = np.log(df['High'] / df['Low']) ** 2
    factor = 1 / (4 * np.log(2))

    df[f'ParkinsonVol_{window}'] = (hl_log.rolling(window).mean() * factor) ** 0.5

    return df
