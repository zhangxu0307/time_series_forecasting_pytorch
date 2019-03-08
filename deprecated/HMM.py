from src import util
from src import eval
import numpy as np
from hmmlearn.hmm import GaussianHMM, GMMHMM

if __name__ == '__main__':

    ts, data = util.load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("../data/pollution.csv", columnName="Ozone")

    train, test = util.divideTrainTest(data)
    print("train shape is", train.shape)
    print("test shape is", test.shape)
    history = [x[0] for x in train]
    predictions = []
    realTestY = []

    for t in range(len(test)):

        model = GaussianHMM(n_components=2)
        model.fit(train)

        output = model.sample(1)

        yhat = output[0][0]

        predictions.append(yhat)
        obs = test[t][0]
        train = np.append(train, obs).reshape(-1, 1)
        realTestY.append(obs)
        print('t:%d, predicted=%f, expected=%f' % (t,  yhat, obs))

    realTestY = np.array(test)
    predictions = np.array(predictions).reshape(-1)
    print("pred:", predictions)
    MAE = eval.calcMAE(realTestY, predictions)
    RMSE = eval.calcRMSE(realTestY, predictions)
    MAPE = eval.calcSMAPE(realTestY, predictions)
    print('Test MAE: %.8f' % MAE)
    print('Test RMSE: %.8f' % RMSE)
    print('Test MAPE: %.8f' % MAPE)

    # plot
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()