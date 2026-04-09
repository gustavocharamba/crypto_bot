import pandas as pd


def get_macd(df, fast=12, slow=26, signal=9):

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df['MACD_pct'] = macd_line / df['Close']
    df['MACD_Signal_pct'] = signal_line / df['Close']
    df['MACD_Hist_pct'] = (macd_line - signal_line) / df['Close']

    return df