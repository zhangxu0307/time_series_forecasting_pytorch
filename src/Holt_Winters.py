
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src import util
from src import eval

# ts, data = util.load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
# ts, data = util.load_data("../data/bike_hour.csv", columnName="cnt")
# ts, data = util.load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
# ts, data = util.load_data("../data/traffic_data_in_bits.csv", columnName="value")
ts, data = util.load_data("../data/beijing_pm25.csv", columnName="pm2.5")
# ts, data = util.load_data("../data/pollution.csv", columnName="Ozone")

train, test = util.divideTrainTest(data)
print("train shape is", train.shape)
print("test shape is", test.shape)

flag = False
lookBack = 48
trainX, trainY = util.createSamples(train, lookBack, RNN=flag)
testX, testY = util.createSamples(test, lookBack, RNN=flag)
print("testX shape:", testX.shape)
print("testy shape:", testY.shape)
print("trainX shape:", trainX.shape)
print("trainy shape:", trainY.shape)

groud_truth = []
prediction = []
for i in range(len(testX)):
    data = testX[i, :].transpose().flatten()
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    model = model.fit(smoothing_level=None)
    pred = model.predict(start=0, end=0)
    real_y = testY[i].tolist()
    groud_truth.extend(pred)
    prediction.extend(real_y)
    print("data:", i)
    print(pred)
    print(real_y)

groud_truth = np.array(groud_truth).reshape(-1, 1)
prediction = np.array(prediction).reshape(-1, 1)
MAE = eval.calcMAE(groud_truth, prediction)
RMSE = eval.calcRMSE(groud_truth, prediction)
SMAPE = eval.calcSMAPE(groud_truth, prediction)
MAPE = eval.calcMAPE(groud_truth, prediction)
print('Test MAE: %.8f' % MAE)
print('Test RMSE: %.8f' % RMSE)
print("test MAPE", MAPE)
print('Test SMAPE: %.8f' % SMAPE)
