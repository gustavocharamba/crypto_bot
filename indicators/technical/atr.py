import pandas as pd
import numpy as np


def get_atr(df, period=14):

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    df['ATR'] = true_range.rolling(window=period).mean()

    df['ATR_Rel'] = df['ATR'] / df['Close']

    return df