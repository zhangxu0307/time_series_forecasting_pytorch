
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


def ts_decompose(ts, freq):

    decomposition = seasonal_decompose(ts.values, model="additive", freq=freq)

    trend = pd.DataFrame(decomposition.trend).dropna().values
    seasonal = pd.DataFrame(decomposition.seasonal).dropna().values
    residual = pd.DataFrame(decomposition.resid).dropna().values
    trend = trend.astype('float32').reshape(-1, 1)
    seasonal = seasonal[freq//2:-(freq//2)].astype('float32').reshape(-1, 1)
    # for unify the scale, drop the head and tail of season series
    residual = residual.astype('float32').reshape(-1, 1)

    return trend, seasonal, residual


if __name__ == "__main__":
    pass