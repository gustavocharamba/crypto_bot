import pandas as pd
import numpy as np


def get_kaufman(df, window=20):

    df = df.copy()

    change = df['Close'].diff(window).abs()
    volatility = df['Close'].diff().abs().rolling(window).sum()

    df[f'ER_{window}'] = change / volatility

    return df
