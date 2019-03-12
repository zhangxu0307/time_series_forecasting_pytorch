'''
machine learning models for time series forecasting
'''

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.linear_model import LinearRegression
from src import eval
from src.util import *
from sklearn.preprocessing import MinMaxScaler
from src.ts_decompose import ts_decompose


# train ML models
def train_ML_model(model, trainX, trainY):

    n = trainX.shape[0]
    print("trainx num is:", n)
    model.fit(trainX, trainY)

    return model


# predict testing set using trained model
def predict_ML_model(model, testX):

    n = testX.shape[0]
    print("testx num is:", n)
    testy = model.predict(testX)

    return testy


# predict testing set for multi-step ahead forecasting, using iteration method
def predict_ML_model_iteration(model, testX, look_ahead):

    testBatchSize = testX.shape[0]

    ans = []

    for i in range(look_ahead):

        pred = model.predict(testX).flatten()  # (test_num, )
        ans.append(pred)

        testX = testX[:, 1:]  # drop the head
        pred = pred.reshape((testBatchSize, 1))  # '1' represents the one-step ahead forecasting
        testX = np.append(testX, pred, axis=1)  # add the prediction results to the tail

    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans


def ML_forecasting(data, model, lag, train_look_ahead, test_look_ahead):

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本
    trainData, testData = divideTrainTest(dataset)

    flag = False
    trainX, trainY = create_multi_ahead_samples(trainData, lag, lookAhead=train_look_ahead, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lag, lookAhead=test_look_ahead, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    model = train_ML_model(model, trainX, trainY)

    testPred = predict_ML_model_iteration(model, testX, test_look_ahead)
    print("testPred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, testY


def decompose_ML_forecasting(model, lag, freq, h_train, h_test):

    # 序列分解
    trend, seasonal, residual = ts_decompose(ts, freq)
    print("ts decomposition is finished!")
    print("trend shape is", trend.shape)
    print("season shape is", seasonal.shape)
    print("residual shape is", residual.shape)

    # forecasting sub-series independently
    trend_pred, trend_y = ML_forecasting(trend, model, lag=lag,  train_look_ahead=h_train, test_look_ahead=h_test)
    res_pred, res_y = ML_forecasting(residual, model, lag=lag,  train_look_ahead=h_train, test_look_ahead=h_test)
    season_pred, season_y = ML_forecasting(seasonal, model, lag=lag,  train_look_ahead=h_train, test_look_ahead=h_test)

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


if __name__ == '__main__':

    # parameters
    lag = 24
    h_train = 1
    h_test = 1
    freq = 8
    print("lag:", lag)
    print("train look ahead:", h_train)
    print("test look ahead:", h_test)
    print("freq:", freq)

    # model
    # model = SVR(C=0.05, epsilon=0.01, kernel="rbf")
    model = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=7)
    # model = DecisionTreeRegressor()
    # model = GaussianHMM(n_components=4)
    # model = LinearRegression()

    # datasets
    ts, data = load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = load_data("../data/pollution.csv", columnName="Ozone")

    # training and testing
    # testPred, testY = ML_forecasting(data, model, lag, train_look_ahead=h_train, test_look_ahead=h_test)
    testPred, testY = decompose_ML_forecasting(model, lag, freq, h_train, h_test)
