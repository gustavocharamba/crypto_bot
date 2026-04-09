import pandas as pd
import numpy as np


def get_skewness(df, window=30):

    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    df['Stat_Skew'] = log_ret.rolling(window=window).skew()

    return df