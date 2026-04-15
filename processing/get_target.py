import pandas as pd
import numpy as np

from processing.pre_processing import get_preprocessing


def get_target(df, window=5, mult_gain=1.5, mult_loss = 0.5 ,horizon=3, spread_past=2, spread_future=2):
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
            future_price = df['Close'].iloc[i + j]

            if future_price >= upper_barrier:
                raw_labels[i] = 1
                break
            elif future_price <= lower_barrier:
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

    df['Stop_Gain_Dyn_Price'] = upper_barriers_dyn
    df['Stop_Loss_Dyn_Price'] = lower_barriers_dyn

    df['Stop_Gain_Pct'] = (np.exp(df['Volatility'] * mult_gain) - 1) * 100
    df['Stop_Loss_Pct'] = (np.exp(-df['Volatility'] * mult_loss) - 1) * 100

    return df


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("../data/btc_1d.csv")
df = get_target(df)
df = get_preprocessing(df)

print(df.tail(10))

