
from pandas import read_csv
import util
import eval
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot
import numpy as np

#ts,dataset = util.load_data(filename,indexName="dteday", columnName="registered")
ts, dataset = util.load_data_xls("./data/NN5/NN5.xlsx", indexName="date", columnName="NN5-003")

# autocorrelation_plot(ts)
# pyplot.show()

X = ts.values
X = np.array(X,dtype="float64")
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(4,1,3))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
#RMSE = np.sqrt(mean_squared_error(test, predictions))
test = np.array(test)
predictions = np.array(predictions).reshape(-1)
MAE = eval.calcMAE(test,predictions)
RMSE = eval.calcRMSE(test,predictions)
MAPE = eval.calcMAPE(test,predictions)
print ('Test MAE: %.8f' % MAE)
print ('Test RMSE: %.8f' % RMSE)
print ('Test MAPE: %.8f' % MAPE)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()