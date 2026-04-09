import pandas as pd

from indicators.technical.macd import get_macd
from indicators.technical.rsi import get_rsi
from indicators.technical.stochastic import get_stoch
from indicators.technical.adx import get_adx
from indicators.technical.atr import get_atr
from indicators.technical.bollinger import get_bollinger
from indicators.technical.obv import get_obv
from indicators.technical.volume import get_volume
from indicators.technical.impulse import get_impulse
from indicators.technical.vwap import get_vwap
from indicators.technical.kaufman import get_kaufman
from indicators.technical.donchian import get_donchian
from indicators.technical.candle_features import get_candle_features
from indicators.technical.time_features import get_time_features
from indicators.technical.historical_levels import get_historical_levels
from indicators.technical.momentum import get_momentum
from indicators.technical.mfi import get_mfi
from indicators.technical.range_expansion import get_range_expansion
from indicators.technical.lagged_candles import get_lagged_candles


def get_technical_indicators(df):

    df = get_macd(df)
    df = get_rsi(df)
    df = get_stoch(df)
    df = get_adx(df)
    df = get_atr(df)
    df = get_bollinger(df)
    df = get_obv(df)
    df = get_volume(df)
    df = get_impulse(df)
    df = get_vwap(df)
    df = get_kaufman(df)
    df = get_donchian(df)
    df = get_candle_features(df)
    df = get_time_features(df)
    df = get_historical_levels(df)
    df = get_momentum(df)
    df = get_mfi(df)
    df = get_range_expansion(df)
    df = get_lagged_candles(df)

    return df