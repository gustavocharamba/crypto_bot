import pandas as pd
import numpy as np

def get_volume(df):
    v_mean = df['Volume'].rolling(window=20).mean()
    v_std = df['Volume'].rolling(window=20).std()

    df['Volume_Z'] = (df['Volume'] - v_mean) / v_std
    df['Volume_Rel'] = df['Volume'] / v_mean

    return df