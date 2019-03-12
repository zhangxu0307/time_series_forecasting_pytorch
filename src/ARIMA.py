from src import util
from src import eval
import numpy as np
import pyflux as pf


if __name__ == '__main__':

    p = 4
    q = 4

    # ts, data = util.load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/traffic_data_in_bits.csv", columnName="value")
    ts, data = util.load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    # ts, data = util.load_data("../data/pollution.csv", columnName="Ozone")

    train, test = util.divideTrainTest(data)
    print("train shape is", train.shape)
    print("test shape is", test.shape)
    history = [x[0] for x in train]
    predictions = []
    realTestY = []

    for t in range(len(test)):

        model = pf.ARIMA(data=np.array(history), ar=p, ma=q, family=pf.Normal())
        model.fit(method="MLE")

        output = model.predict(1, intervals=False)

        yhat = output.values[0][0]

        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        realTestY.append(obs)
        print('t:%d, predicted=%f, expected=%f' % (t,  yhat, obs))

    realTestY = np.array(test).reshape(-1, 1)
    predictions = np.array(predictions).reshape(-1, 1)
    print("pred:", predictions)
    MAE = eval.calcMAE(realTestY, predictions)
    RMSE = eval.calcRMSE(realTestY, predictions)
    MAPE = eval.calcSMAPE(realTestY, predictions)
    print('Test MAE: %.8f' % MAE)
    print('Test RMSE: %.8f' % RMSE)
    print('Test SMAPE: %.8f' % MAPE)

    # plot
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()