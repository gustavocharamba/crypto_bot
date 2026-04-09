import pandas as pd
import numpy as np

def get_donchian(df, period=20):

    high = df['High'].rolling(period).max()
    low = df['Low'].rolling(period).min()
    df[f'DonchianWidth_{period}'] = (high - low) / df['Close']

    return df
