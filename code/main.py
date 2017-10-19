from code.train import train, predict
from code.util import load_data, load_data_txt, load_data_xls, createSamples, divideTrainTest, align
from sklearn.preprocessing import MinMaxScaler
from code import eval
from code.season_decompose import seasonDecompose
import pandas as pd

def test(data, lookBack, epoch, lr, batchSize, method, modelPath):


    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本,并整理成RNN的输入形式
    trainData, testData = divideTrainTest(dataset)

    flag = True
    trainX, trainY = createSamples(trainData, lookBack, RNN=flag)
    testX, testY = createSamples(testData, lookBack, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)


    train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batchSize, modelPath=modelPath,
          lookBack=lookBack, method=method)

    testPred = predict(testX, MODEL_PATH)
    trainPred = predict(trainX, MODEL_PATH)
    print("testPred shape:", testPred.shape)
    print("trainPred shape:", trainPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    # MAPE = eval.calcMAPE(testY, testPred)
    # print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return trainPred, testPred, MAE, MRSE, SMAPE


def FCD_Train(ts, dataset, freq, lookBack, batchSize, epoch, lr, method):


    # 序列分解
    #ts.index = pd.date_range(start='19960318',periods=len(ts), freq='Q')
    trend, seasonal, residual = seasonDecompose(ts, freq)
    print("fcd decompose is finised!")
    print("trend shape is", trend.shape)
    print("season shape is", seasonal.shape)
    print("residual shape is", residual.shape)

    # 分别预测

    MODEL_PATH = "../model/ResRNN_model.pkl"
    # trainPred, testPred, MAE, MRSE, SMAPE = test(data=dataset, lookBack=lag, epoch=epoch,
    #                                              batchSize=batchSize, method=method, modelPath=MODEL_PATH)
    trTrain, trTest, MAE1, MRSE1, SMAPE1 = test(trend, lookBack, epoch, lr, batchSize,  method=method, modelPath=MODEL_PATH)
    resTrain, resTest, MAE2, MRSE2, SMAPE2 = test(residual, lookBack, epoch, lr, batchSize, method=method, modelPath=MODEL_PATH)
    # trTrain, trTest, MAE1, MRSE1, SMAPE1= RNNFORECAST.RNNforecasting(trend, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4,
    #                                                                     hiddenNum=100)
    # resTrain, resTest, MAE2, MRSE2, SMAPE2 = RNNFORECAST.RNNforecasting(residual, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4, hiddenNum=100)

    trTrain = trTrain.reshape(-1)
    trTest = trTest.reshape(-1)
    resTrain = resTrain.reshape(-1)
    resTest = resTest.reshape(-1)

    print("trTrain shape is", trTrain.shape)
    print("resTrain shape is", resTrain.shape)

    # '''
    # 数据对齐
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

    # plt.plot(data)
    # plt.plot(finalPred)
    # plt.show()
    # '''
    return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == "__main__":

    lookBack = 24
    batchSize = 30
    epoch = 2
    MODEL_PATH = "../model/ResRNN_model.pkl"
    METHOD = "ResRNN"
    freq = 48
    lr = 1e-4

    print("looback:", lookBack)
    print("batchSize", batchSize)
    print("epoch:", epoch)
    print("METHOD:", METHOD)
    print("freq:", freq)
    print("MODEL_PATH:", MODEL_PATH)
    print("lr:", lr)

    ts, data = load_data("../data/NSW2013.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    #ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
    #ts, data = load_data("../data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
    #ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    trainPred, testPred, MAE, MRSE, SMAPE = test(data=data, lookBack=lookBack, epoch=epoch,  lr=lr,
                                                batchSize=batchSize, method=METHOD, modelPath=MODEL_PATH)

    # trainPred, testPred, MAE, MRSE, SMAPE = FCD_Train(ts=ts, dataset=data, freq=freq, lookBack=lookBack,
    #                                                   batchSize=batchSize,
    #                                                   epoch=epoch, lr=lr, method=METHOD)

    #test(data, lookBack, epoch, 1e-4, batchSize,  method=METHOD, modelPath=MODEL_PATH)