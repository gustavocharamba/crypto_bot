import pandas as pd
import numpy as np

def get_vwap(df, window=20):

    df = df.copy()

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (tp * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()

    df[f'VWAP_dist_{window}'] = (df['Close'] - vwap) / vwap

    return df
