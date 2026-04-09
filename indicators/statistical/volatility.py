import pandas as pd
import numpy as np


def get_volatility(df, window=30):

    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    df['Log_Ret'] = log_ret
    df['Stat_Vol'] = log_ret.rolling(window=window).std()

    return df