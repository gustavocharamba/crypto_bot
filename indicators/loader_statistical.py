from indicators.statistical.zscore import get_zscore
from indicators.statistical.skewness import get_skewness
from indicators.statistical.kurtosis import get_kurtosis
from indicators.statistical.parkinson import get_parkinson
from indicators.statistical.autocorr import get_autocorr
from indicators.statistical.vol_regime import get_vol_regime
# Stat_Vol removida: substituída por Parkinson Vol (superior), ATR,
# range_expansion e vol_regime — todas cobrem volatilidade de forma mais rica


def get_statistical_indicators(df):

    df = get_zscore(df)
    df = get_skewness(df)
    df = get_kurtosis(df)
    df = get_parkinson(df)
    df = get_autocorr(df)
    df = get_vol_regime(df)

    return df