from sklearn.svm import SVR
from code import eval
from code.util import *
from code.season_decompose import *
from sklearn.preprocessing import MinMaxScaler


def trainSVM(trainX, trainY):

    n = trainX.shape[0]
    print("trainx num is:", n)
    svrModel = SVR(C=0.1, epsilon=0.01, kernel="rbf")
    svrModel.fit(trainX, trainY)

    return svrModel

def predictSVM(testX, svrModel):

    n = testX.shape[0]
    print("testx num is:", n)
    testy = svrModel.predict(testX)

    return testy


def testSVM(data, lookBack):

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本
    trainData, testData = divideTrainTest(dataset)

    flag = False
    trainX, trainY = createSamples(trainData, lookBack, RNN=flag)
    testX, testY = createSamples(testData, lookBack, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    model = trainSVM(trainX, trainY)

    testPred = predictSVM(testX, model)
    trainPred = predictSVM(trainX, model)
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


def FCD_Train_SVM(ts, dataset, freq, lookBack):


    # 序列分解
    #ts.index = pd.date_range(start='19960318',periods=len(ts), freq='Q')
    trend, seasonal, residual = seasonDecompose(ts, freq)
    print("fcd decompose is finised!")
    print("trend shape is", trend.shape)
    print("season shape is", seasonal.shape)
    print("residual shape is", residual.shape)

    # 分别预测

    trTrain, trTest, MAE1, MRSE1, SMAPE1 = testSVM(trend, lookBack)
    resTrain, resTest, MAE2, MRSE2, SMAPE2 = testSVM(residual, lookBack)

    trTrain = trTrain.reshape(-1)
    trTest = trTest.reshape(-1)
    resTrain = resTrain.reshape(-1)
    resTest = resTest.reshape(-1)

    print("trTrain shape is", trTrain.shape)
    print("resTrain shape is", resTrain.shape)

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


if __name__ == '__main__':

    lookBack = 24
    freq = 8

    print("looback:", lookBack)
    print("freq:", freq)

    ts, data = load_data("../data/NSW2013.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
    #ts, data = load_data("../data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
    #ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    trainPred, testPred, MAE, MRSE, SMAPE = testSVM(data, lookBack)

    #trainPred, testPred, MAE, MRSE, SMAPE = FCD_Train_SVM(ts, data, freq, lookBack)
