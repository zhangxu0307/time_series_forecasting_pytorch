from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.linear_model import LinearRegression
from src import  eval
from src.util import *
from sklearn.preprocessing import MinMaxScaler


def train_ML_model(model, trainX, trainY):

    n = trainX.shape[0]
    print("trainx num is:", n)
    model.fit(trainX, trainY)

    return model


def predict_ML_model(testX, model):

    n = testX.shape[0]
    print("testx num is:", n)
    testy = model.predict(testX)

    return testy


def predict_ML_model_iteration(testX, lookAhead, model):

    testBatchSize = testX.shape[0]
    ans = []

    for i in range(lookAhead):

        pred = model.predict(testX).flatten()  # (test_num, )
        ans.append(pred)

        testX = testX[:, 1:]
        pred = pred.reshape((testBatchSize, 1))
        testX = np.append(testX, pred, axis=1)

    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans


def ML_forecasting(data, model, lookBack, train_lookAhead, test_lookAhead):

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    # 分割序列为样本
    trainData, testData = divideTrainTest(dataset)

    flag = False
    trainX, trainY = create_multi_ahead_samples(trainData, lookBack, lookAhead=train_lookAhead, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lookBack, lookAhead=test_lookAhead, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    model = train_ML_model(model, trainX, trainY)

    testPred = predict_ML_model_iteration(testX, test_lookAhead, model)
    print("testPred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    MAE = eval.calcMAE(testY, testPred)
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", MRSE)
    SMAPE = eval.calcSMAPE(testY, testPred)
    print("test SMAPE", SMAPE)

    return testPred, MAE, MRSE, SMAPE


if __name__ == '__main__':

    lookBack = 24
    train_lookAhead = 1
    test_lookAhead = 1

    # model = SVR(C=0.05, epsilon=0.01, kernel="rbf")
    model = RandomForestRegressor(n_estimators=30, max_depth=5, n_jobs=7)
    # model = DecisionTreeRegressor()
    # model = GaussianHMM(n_components=4)
    # model = LinearRegression()

    print("looback:", lookBack)
    print("train look ahead:", train_lookAhead)
    print("test look ahead:", test_lookAhead)

    # ts, data = load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    ts, data = load_data("../data/pollution.csv", columnName="Ozone")

    testPred, MAE, MRSE, SMAPE = ML_forecasting(data, model, lookBack, train_lookAhead=train_lookAhead, test_lookAhead=test_lookAhead)

