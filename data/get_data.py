import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"

start = int(pd.Timestamp("2021-01-01").timestamp() * 1000)
end   = int(pd.Timestamp("2026-01-01").timestamp() * 1000)

limit_candles = 1000

intervals = ["30m", "1h", "2h", "4h", "1d"]


def get_data(interval, start_time, end_time):
    data = []

    while start_time < end_time:
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit_candles
        }

        r = requests.get(BASE_URL, params=params)
        klines = r.json()

        if len(klines) == 0:
            break

        data.extend(klines)
        start_time = klines[-1][0] + 1

        print(f"{interval} candles: {len(data)}")
        time.sleep(0.1)

    cols = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades",
        "taker_base", "taker_quote", "ignore"
    ]

    df = pd.DataFrame(data, columns=cols)
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })

    df = df[["time", "open", "high", "low", "close", "volume"]]
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

    df = df.drop_duplicates()
    df = df.sort_values("Datetime")

    return df

for interval in intervals:
    df = get_data(interval, start, end)
    filename = f"btc_{interval}.csv"
    df.to_csv(filename, index=False)

    print(f"Arquivo salvo: {filename}")