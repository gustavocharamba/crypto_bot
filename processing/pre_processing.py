import pandas as pd


def get_preprocessing(df):
    df = df.copy()

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols + ['Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Volume'] = df['Volume'].fillna(0.0)
    df.dropna(subset=price_cols, inplace=True)

    return df
