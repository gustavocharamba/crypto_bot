import pandas as pd
import numpy as np


def get_target(df, window=5, mult_gain=1.5, mult_loss=0.5, horizon=3, spread_past=2, spread_future=2):
    """
    Cria um alvo binário para região de compra.

    Raw_Target:
        1  = gain tocado antes do stop dentro do horizonte
       -1  = stop tocado antes do gain dentro do horizonte
        0  = nenhuma barreira tocada

    Target:
        1 = região de compra ao redor de um Raw_Target positivo
        0 = demais regiões
    """
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Ret'].rolling(window=window).std()

    raw_labels = np.zeros(len(df))
    upper_barriers_dyn = np.full(len(df), np.nan)
    lower_barriers_dyn = np.full(len(df), np.nan)

    for i in range(len(df) - horizon):
        if pd.isna(df['Volatility'].iloc[i]):
            continue

        p0 = df['Close'].iloc[i]
        vol = df['Volatility'].iloc[i]

        upper_barrier = p0 * np.exp(vol * mult_gain)
        lower_barrier = p0 * np.exp(-vol * mult_loss)

        upper_barriers_dyn[i] = upper_barrier
        lower_barriers_dyn[i] = lower_barrier

        for j in range(1, horizon + 1):
            future_high = df['High'].iloc[i + j]
            future_low = df['Low'].iloc[i + j]

            hit_gain = future_high >= upper_barrier
            hit_loss = future_low <= lower_barrier

            if hit_gain and hit_loss:
                raw_labels[i] = -1
                break
            elif hit_gain:
                raw_labels[i] = 1
                break
            elif hit_loss:
                raw_labels[i] = -1
                break

    df['Raw_Target'] = raw_labels

    shifted_Targets = [df['Raw_Target']]
    for p in range(1, spread_past + 1):
        shifted_Targets.append(df['Raw_Target'].shift(-p))
    for f in range(1, spread_future + 1):
        shifted_Targets.append(df['Raw_Target'].shift(f))

    df['Target'] = pd.concat(shifted_Targets, axis=1).max(axis=1)

    shifted_mins = [df['Raw_Target']]
    for p in range(1, spread_past + 1):
        shifted_mins.append(df['Raw_Target'].shift(-p))
    for f in range(1, spread_future + 1):
        shifted_mins.append(df['Raw_Target'].shift(f))

    mask_neg = pd.concat(shifted_mins, axis=1).min(axis=1)
    df['Target'] = np.where(mask_neg == -1, -1, df['Target'])
    df['Target'] = (df['Target'] == 1).astype(int)

    df['Stop_Gain_Dyn_Price'] = upper_barriers_dyn
    df['Stop_Loss_Dyn_Price'] = lower_barriers_dyn

    df['Stop_Gain_Pct'] = (np.exp(df['Volatility'] * mult_gain) - 1) * 100
    df['Stop_Loss_Pct'] = (np.exp(-df['Volatility'] * mult_loss) - 1) * 100

    return df
