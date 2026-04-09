def get_impulse(df):

    # BB_Width: contração/expansão das Bandas de Bollinger
    std    = df['Close'].rolling(window=20).std()
    middle = df['Close'].rolling(window=20).mean()
    upper  = middle + (std * 2)
    lower  = middle - (std * 2)
    df['BB_Width'] = (upper - lower) / middle

    # Volume_Z: anomalia de volume normalizada
    v_mean = df['Volume'].rolling(window=20).mean()
    v_std  = df['Volume'].rolling(window=20).std()
    df['Volume_Z'] = (df['Volume'] - v_mean) / v_std

    return df