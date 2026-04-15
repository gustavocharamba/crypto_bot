import pandas as pd

def get_preprocessing(df):

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Volume'] = df['Volume'].fillna(0).astype(int)

    return df

