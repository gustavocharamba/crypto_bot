import pandas as pd
import numpy as np


def get_target(df, window=20, multiplier=2, horizon=5, spread_past=2, spread_future=1):
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol'] = df['log_ret'].rolling(window=window).std()

    raw_labels = np.zeros(len(df))

    for i in range(len(df) - horizon):
        if pd.isna(df['vol'].iloc[i]):
            continue

        p0 = df['close'].iloc[i]
        vol = df['vol'].iloc[i]

        upper_barrier = p0 * np.exp(vol * multiplier)
        lower_barrier = p0 * np.exp(-vol * multiplier)

        for j in range(1, horizon + 1):
            future_price = df['close'].iloc[i + j]

            if future_price >= upper_barrier:
                raw_labels[i] = 1
                break
            elif future_price <= lower_barrier:
                raw_labels[i] = -1
                break

    df['raw_target'] = raw_labels

    shifted_targets = [df['raw_target']]
    for p in range(1, spread_past + 1):
        shifted_targets.append(df['raw_target'].shift(-p))
    for f in range(1, spread_future + 1):
        shifted_targets.append(df['raw_target'].shift(f))

    df['target'] = pd.concat(shifted_targets, axis=1).max(axis=1)

    shifted_mins = [df['raw_target']]
    for p in range(1, spread_past + 1):
        shifted_mins.append(df['raw_target'].shift(-p))
    for f in range(1, spread_future + 1):
        shifted_mins.append(df['raw_target'].shift(f))

    mask_neg = pd.concat(shifted_mins, axis=1).min(axis=1)

    df['target'] = np.where(mask_neg == -1, -1, df['target'])

    return df