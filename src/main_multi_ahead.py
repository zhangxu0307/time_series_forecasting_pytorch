from src.train import train, predict, predict_iteration
from src.util import *
from sklearn.preprocessing import MinMaxScaler
from src import eval
from src.ts_decompose import ts_decompose
import pandas as pd


def single_model_forecasting(data, lookBack, epoch, h_train, h_test, lr, batchSize,
                             method, modelPath, hidden_num, use_cuda):

    # 归一化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    trainData, testData = divideTrainTest(dataset)

    flag = True
    trainX, trainY = create_multi_ahead_samples(trainData, lookBack, h_train, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lookBack, h_test, RNN=flag)
    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)


    net = train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batchSize, modelPath=modelPath,
          lookBack=lookBack, method=method, hidden_num=hidden_num, use_cuda=use_cuda)

    testPred = predict_iteration(net, testX, h_test, use_cuda=use_cuda, RNN=flag)
    trainPred = predict_iteration(net, trainX, h_train, use_cuda=use_cuda, RNN=flag)
    print("train pred shape:", trainPred.shape)
    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return trainPred, testPred, MAE, MRSE, SMAPE


def decomposition_model_forecasting(ts, dataset, lookBack, freq, epoch, lr, batchSize, hidden_num, use_cuda):

    # 序列分解
    trend, seasonal, residual = ts_decompose(ts, freq)
    print("fcd decompose is finised!")
    print("trend shape is", trend.shape)
    print("season shape is", seasonal.shape)
    print("residual shape is", residual.shape)

    # 分别预测

    MODEL_PATH = "../models/ResRNN_model.pkl"
    trTrain, trTest, MAE1, MRSE1, SMAPE1 = single_model_forecasting(trend, lookBack=lag, epoch=epoch,  lr=lr, use_cuda=use_cuda,
                                                batchSize=batchSize, method=METHOD, modelPath=MODEL_PATH, hidden_num=hidden_num)
    resTrain, resTest, MAE2, MRSE2, SMAPE2 = single_model_forecasting(residual, lookBack=lag, epoch=epoch,  lr=lr, use_cuda=use_cuda,
                                                batchSize=batchSize, method=METHOD, modelPath=MODEL_PATH, hidden_num=hidden_num)

    trTrain = trTrain.reshape(-1, 1)
    trTest = trTest.reshape(-1, 1)
    resTrain = resTrain.reshape(-1, 1)
    resTest = resTest.reshape(-1, 1)

    print("trTrain shape is", trTrain.shape)
    print("resTrain shape is", resTrain.shape)

    trendPred, resPred = align(trTrain, trTest, lookBack, resTrain, resTest, lookBack)

    print("trendPred shape is", trendPred.shape)
    print("resPred shape is", resPred.shape)

    # 获取最终预测结果
    finalPred = trendPred + seasonal + resPred

    trainPred = trTrain + seasonal[lookBack:lookBack + trTrain.shape[0]] + resTrain
    testPred = trTest + seasonal[2 * lookBack + resTrain.shape[0] :] + resTest

    # 获得ground-truth数据
    data = dataset[freq // 2:-(freq // 2)]
    trainY = data[lookBack:lookBack + trTrain.shape[0]]
    testY = data[2 * lookBack + resTrain.shape[0]:]
    print(trainY.shape)
    print(testY.shape)
    print(trainPred.shape)
    print(testPred.shape)

    # 评估指标
    MAE = eval.calcMAE(trainY, trainPred)
    print("train MAE", MAE)
    MRSE = eval.calcRMSE(trainY, trainPred)
    print("train MRSE", MRSE)
    MAPE = eval.calcMAPE(trainY, trainPred)
    print("train MAPE", MAPE)
    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(testY, testPred)
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return trainPred, testPred, MAE, MRSE, SMAPE


if __name__ == "__main__":

    lag = 24
    h_train = 1
    h_test = 6
    batchSize = 32
    epoch = 20
    MODEL_PATH = "../models/RNN_model.pkl"
    METHOD = "ResRNN"
    freq = 4
    lr = 1e-4
    hidden_num = 64

    print("lag:", lag)
    print("batchSize", batchSize)
    print("h train:", h_train)
    print("h test:", h_test)
    print("epoch:", epoch)
    print("METHOD:", METHOD)
    print("freq:", freq)
    print("MODEL_PATH:", MODEL_PATH)
    print("lr:", lr)

    # ts, data = load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    ts, data = load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = load_data("../data/pollution.csv", columnName="Ozone")

    trainPred, testPred, MAE, MRSE, SMAPE = single_model_forecasting(data=data, lookBack=lag, h_train=h_train, h_test=h_test,
                                                                     epoch=epoch,  lr=lr, use_cuda=True, batchSize=batchSize,
                                                                     method = METHOD, modelPath=MODEL_PATH, hidden_num=hidden_num)

    # trainPred, testPred, MAE, MRSE, SMAPE = decomposition_model_forecasting(ts, data, lag,
    #                                  freq, epoch, lr, batchSize, hidden_num, use_cuda=True)
