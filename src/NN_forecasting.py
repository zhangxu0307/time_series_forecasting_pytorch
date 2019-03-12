'''
Neural Networks models for time series forecasting
'''

from src.NN_train import train, predict, predict_iteration
from src.util import *
from sklearn.preprocessing import MinMaxScaler
from src import eval
from src.ts_decompose import ts_decompose
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# forecasting using neural networks, support multi-step ahead forecasting
def single_model_forecasting(data, lag, h_train, h_test, lr, epoch, batch_size, hidden_num, method, use_cuda):

    # normalize time series
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    trainData, testData = divideTrainTest(dataset)

    flag = True  # using RNN format or not
    trainX, trainY = create_multi_ahead_samples(trainData, lag, h_train, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lag, h_test, RNN=flag)
    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    print("train X shape:", trainX.shape)
    print("train y shape:", trainY.shape)
    print("test X shape:", testX.shape)
    print("test y shape:", testY.shape)

    net = train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batch_size,
                lag=lag, method=method, hidden_num=hidden_num, use_cuda=use_cuda)

    testPred = predict_iteration(net, testX, h_test, use_cuda=use_cuda, RNN=flag)
    # trainPred = predict_iteration(net, trainX, h_train, use_cuda=use_cuda, RNN=flag)
    # print("train pred shape:", trainPred.shape)
    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluation
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, testY


# forecasting using neural networks with time series decomposition, support multi-step ahead forecasting
def decomposition_model_forecasting(ts, dataset, lag, h_train, h_test, freq, epoch, lr, batch_size, hidden_num, use_cuda, method):

    # time series decomposition
    trend, seasonal, residual = ts_decompose(ts, freq)
    print("ts decomposition is finished!")
    print("trend shape is", trend.shape)
    print("season shape is", seasonal.shape)
    print("residual shape is", residual.shape)

    # forecasting sub-series independently

    trend_pred, trend_y = single_model_forecasting(trend, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                                   hidden_num=hidden_num, batch_size=batch_size, method=method, use_cuda=use_cuda)

    res_pred, res_y = single_model_forecasting(residual, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                               hidden_num=hidden_num, batch_size=batch_size, method=method, use_cuda=use_cuda)

    season_pred, season_y = single_model_forecasting(seasonal, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                                     hidden_num=hidden_num, batch_size=batch_size, method=method, use_cuda=use_cuda)

    trend_pred = trend_pred.reshape(-1, h_test)
    trend_y = trend_y.reshape(-1, h_test)
    res_pred = res_pred.reshape(-1, h_test)
    res_y = res_y.reshape(-1, h_test)
    season_pred = season_pred.reshape(-1, h_test)
    season_y = season_y.reshape(-1, h_test)

    print("trend_pred shape is", trend_pred.shape)
    print("res_pred shape is", res_pred.shape)
    print("season_pred shape is", season_pred.shape)
    print("trend_y shape is", trend_y.shape)
    print("res_y shape is", res_y.shape)
    print("season_y shape is", season_y.shape)

    testPred = trend_pred + res_pred + season_pred
    testY = trend_y + res_y + season_y

    # evaluation
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, testY


if __name__ == "__main__":

    # parameters
    lag = 24
    h_train = 1
    h_test = 1
    batch_size = 32
    epoch = 20
    METHOD = "RNN"
    freq = 8
    lr = 1e-4
    hidden_num = 64

    print("lag:", lag)
    print("batch size", batch_size)
    print("h train:", h_train)
    print("h test:", h_test)
    print("epoch:", epoch)
    print("METHOD:", METHOD)
    print("freq:", freq)
    print("lr:", lr)

    # datasets
    ts, data = load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = load_data("../data/pollution.csv", columnName="Ozone")

    # training and testing
    # testPred, testY = single_model_forecasting(data=data, lag=lag, h_train=h_train, h_test=h_test,
    #                                            epoch=epoch,  lr=lr, use_cuda=True, batch_size=batch_size,
    #                                            method=METHOD, hidden_num=hidden_num)

    testPred, testY = decomposition_model_forecasting(ts=ts, dataset=data, lag=lag, h_train=h_train,  h_test=h_test,
                                                      epoch=epoch,  lr=lr, use_cuda=True, batch_size=batch_size, freq=freq,
                                                      method=METHOD, hidden_num=hidden_num)
