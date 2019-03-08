#encoding=utf-8
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np


# 计算RMSE
def calcRMSE(true,pred):
    return np.sqrt(mean_squared_error(true, pred))


# 计算MAE
def calcMAE(true,pred):
    return mean_absolute_error(true, pred)


# 计算MAPE
def calcMAPE(true, pred, epsion = 0.0000000):

    true += epsion
    return np.mean(np.abs((true-pred)/true))*100


# 计算SMAPE
def calcSMAPE(true, pred):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.mean(np.abs((true-pred)/delim))*100