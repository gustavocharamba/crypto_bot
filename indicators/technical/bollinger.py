import pandas as pd
import numpy as np


def get_bollinger(df, period=20, std_dev=2):

    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    df['BB_Percent_B'] = (df['Close'] - lower) / (upper - lower)
    df['BB_Width'] = (upper - lower) / middle
    df['BB_Middle_pct'] = (df['Close'] - middle) / middle

    return df