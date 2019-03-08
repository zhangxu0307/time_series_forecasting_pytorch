# time\_series\_forecasting_pytorch
Experimental source code:
Time series forecasting using pytorch，including MLP,RNN,LSTM,GRU, ARIMA, SVR, RF and TSR-RNN models.

Requirements

- python 3.6.3 (Anaconda)
- keras 2.1.2
- PyTorch 1.0.1
- tensorflow-gpu 1.13.1
- sklearn 0.19.1
- numpy 1.15.4
- pandas 0.23.4
- statsmodels 0.8.0
- matplotlib 2.1.0

## Code
- ARIMA.py: ARIMA model, iteration version
- eval.py: evaluation metrics, including RMSE,MAE,MAPE and SMAPE.
- main.py: run one-step-ahead forecasting
- main\_multi_ahead.py:run multi-step-ahead forecasting
- model.py: neural network models
- train.py: training of neural network models
- ts_decompose.py: time series decomposition
- ts_loader: data loader for neural network models
- ML_forecasting.py: general machine learning models, including SVR and RF
- util.py: data loader

## Related Repository

[Time Series Forecasting using Keras](https://github.com/zhangxu0307/time-series-forecasting-keras)