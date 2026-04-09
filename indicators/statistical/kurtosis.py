import pandas as pd
import numpy as np


def get_kurtosis(df, window=30):

    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    df['Stat_Kurt'] = log_ret.rolling(window=window).kurt()

    return df