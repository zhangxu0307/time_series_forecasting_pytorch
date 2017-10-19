from code.season_decompose import seasonDecompose
from code.train import train, predict
from code.util import load_data, load_data_txt, load_data_xls, createSamples, divideTrainTest, align
from sklearn.preprocessing import MinMaxScaler
from code import eval
from code.main import test
import time


def FCD_Train(ts, dataset, freq, lag, batchSize, epoch, method):


    # 序列分解
    # ts.index = pd.date_range(start='19960318',periods=len(ts), freq='Q')
    # trend, seasonal, residual = seasonDecompose(ts, freq)
    # print("fcd decompose is finised!")
    # print("trend shape is", trend.shape)
    # print("season shape is", seasonal.shape)
    # print("residual shape is", residual.shape)
    #
    # # 分别预测
    # t1 = time.time()

    MODEL_PATH = "../model/ResRNN_model.pkl"
    trainPred, testPred, MAE, MRSE, SMAPE = test(data=dataset, lookBack=lag, epoch=epoch,
                                                 batchSize=batchSize, method=method, modelPath=MODEL_PATH)
    #trTrain, trTest, MAE1, MRSE1, SMAPE1 = test(dataset, lookBack, epoch, batchSize, method="RNN", modelPath=MODEL_PATH)
    #resTrain, resTest, MAE2, MRSE2, SMAPE2 = test(residual, lookBack, epoch, batchSize, method="RNN", modelPath=MODEL_PATH)
    # trTrain, trTest, MAE1, MRSE1, SMAPE1= RNNFORECAST.RNNforecasting(trend, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4,
    #                                                                     hiddenNum=100)
    # resTrain, resTest, MAE2, MRSE2, SMAPE2 = RNNFORECAST.RNNforecasting(residual, lookBack=resWin, epoch=30, unit=unit,
    #                                                                     varFlag=True, minLen=20, maxLen=lag, step=4, hiddenNum=100)
    t2 = time.time()
    # print(t2 - t1)
    #
    # print("trTrain shape is", trTrain.shape)
    # print("resTrain shape is", resTrain.shape)
    #
    # # '''
    # # 数据对齐
    # trendPred, resPred = align(trTrain, trTest, lookBack, resTrain, resTest, lookBack)
    #
    # print("trendPred shape is", trendPred.shape)
    # print("resPred shape is", resPred.shape)
    #
    # # 获取最终预测结果
    # finalPred = trendPred + seasonal + resPred
    #
    # trainPred = trTrain + seasonal[lookBack:lookBack + trTrain.shape[0]] + resTrain
    # testPred = trTest + seasonal[2 * lookBack + resTrain.shape[0] :] + resTest
    #
    # # 获得ground-truth数据
    # data = dataset[freq // 2:-(freq // 2)]
    # trainY = data[lookBack:lookBack + trTrain.shape[0]]
    # testY = data[2 * lookBack + resTrain.shape[0]:]
    #
    # # 评估指标
    # MAE = eval.calcMAE(trainY, trainPred)
    # print("train MAE", MAE)
    # MRSE = eval.calcRMSE(trainY, trainPred)
    # print("train MRSE", MRSE)
    # MAPE = eval.calcMAPE(trainY, trainPred)
    # print("train MAPE", MAPE)
    # MAE = eval.calcMAE(testY, testPred)
    # print("test MAE", MAE)
    # MRSE = eval.calcRMSE(testY, testPred)
    # print("test RMSE", MRSE)
    # MAPE = eval.calcMAPE(testY, testPred)
    # print("test MAPE", MAPE)
    # SMAPE = eval.calcSMAPE(testY, testPred)
    # print("test SMAPE", SMAPE)
    #
    # # plt.plot(data)
    # # plt.plot(finalPred)
    # # plt.show()
    # # '''
    # return trainPred, testPred, MAE, MRSE, SMAPE

if __name__ == '__main__':

    lookBack = 24
    batchSize = 30
    epoch = 20
    MODEL_PATH = "../model/RNN_model.pkl"
    METHOD = "RNN"
    freq = 4

    ts, data = load_data("../data/NSW2013.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", indexName="dteday", columnName="cnt")
    #ts, data = load_data("../data/traffic_data_in_bits.csv", indexName="datetime", columnName="value")
    # ts, data = load_data("../data/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")

    trainPred, testPred, MAE, MRSE, SMAPE = FCD_Train(ts=ts, dataset=data, freq=freq, lag=lookBack, batchSize=batchSize, epoch=epoch, method=METHOD)