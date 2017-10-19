#encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.diagnostic
import statsmodels.api as sm

# 加载数据，返回时间序列series和array形式的数据,以行形式返回
def load_data(filename, columnName, indexName):

    df = pd.read_csv(filename)
    #df.index = pd.to_datetime(df.index)
    ts = df[columnName]
    data = ts.values.reshape(-1).astype("float32")
    return ts, data

# 加载excel文件数据，选定时间轴和要预测的列名称
def load_data_xls(filename, indexName, columnName):

    df = pd.read_excel(filename, index_col=indexName)
    df.index = pd.to_datetime(df.index)
    df = df.fillna(method='pad')
    ts = df[columnName]
    data = pd.DataFrame(ts).values.reshape(-1)
    return ts, data

# 加载txt数据
def load_data_txt(filename, indexName, columnName):

    reader = pd.read_table(filename,header=0,index_col=indexName,delimiter=";", iterator=True)
    df = reader.get_chunk(5000)
    df.index = pd.to_datetime(df.index)
    #df = df.fillna(method='pad')
    ts = df[columnName]
    data = pd.DataFrame(ts).values.reshape(-1)
    return ts, data

# 分割时间序列作为样本，lookBack为窗口大小,并整理成RNN的输入形式
def createSamples(dataset,lookBack,RNN = True):

    dataX, dataY = [], []
    for i in range(len(dataset) - lookBack):
        a = dataset[i:(i + lookBack)]
        dataX.append(a)
        dataY.append(dataset[i + lookBack])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    if RNN:
        dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
    return dataX, dataY


# 划分训练集和测试集，除NN5按9:1外，其余均按照3:1划分
def divideTrainTest(dataset,rate = 0.75):

    train_size = int(len(dataset) * rate)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:]
    return train,test

# 变长度采样训练样本
'''
def createVariableDataset(dataset, minLen, maxLen,numInputs):
    dataX = []
    dataY = []
    for i in range(numInputs):
        start = np.random.randint(len(dataset)-minLen-1)
        #end = np.random.randint(min(start+minLen, len(dataset)-1), min(start+maxLen, len(dataset)-1))
        randomLen = np.random.randint(minLen, maxLen)
        if start+randomLen > len(dataset)-2:
            end = len(dataset)-2
        else:
            end = start+randomLen
        sequence_in = dataset[start:end]
        #sequence_in = pad_sequences(sequence_in, maxlen=maxLen, dtype='float32')
        sequence_out = dataset[end + 1]
        dataX.append(sequence_in)
        dataY.append(sequence_out)
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = pad_sequences(dataX, maxlen=maxLen, dtype='float32') # 左端补齐
    dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1)) # 转化为rnn输入形式
    return dataX,dataY
'''

# # 取不同长度的子序列，以重点为基准点向前搜索出lag区间段的所有子序列
# def createVariableDataset(dataset, minLen, maxLen, step):
#
#     dataNum = len(dataset)
#     X,Y = [],[]
#
#     for i in range(maxLen, len(dataset)):
#         for lookBack in range(minLen, maxLen + 1, step):  # 遍历所有长度
#             a = dataset[i-lookBack:i]
#             X.append(a)
#             Y.append(dataset[i])
#     X = np.array(X)
#     Y = np.array(Y)
#     X = pad_sequences(X, maxlen=maxLen, dtype='float32')  # 左端补齐
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#     return X, Y
#
# # 将测试用的ground-truth转化为标准形式，与上面的函数一起使用
# def transformGroundTruth(vtestY, minLen, maxLen, step):
#
#     lagNum = (maxLen - minLen)//step+1
#     print("lag num is", lagNum)
#     truth = []
#     for i in range(0, len(vtestY), lagNum):
#         truth.append(np.mean(vtestY[i:i + lagNum]))
#     return np.array(truth)
#
#
# # 取不同长度样本并补足成最大长度
# def createPaddedDataset(dataset, lookBack, maxLen):
#
#     dataX, dataY = [], []
#     for i in range(len(dataset) - lookBack):
#         a = dataset[i:(i + lookBack)]
#         dataX.append(a)
#         dataY.append(dataset[i + lookBack])
#     dataX = np.array(dataX)
#     dataY = np.array(dataY)
#     dataX = pad_sequences(dataX, maxlen=maxLen, dtype='float32')  # 左端补齐
#     dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
#     return dataX, dataY



# 数据对齐
def align(trTrain,trTest,trendWin,resTrain,resTest,resWin):

    empWin = np.empty((trendWin))
    empWin[:] = np.nan

    empWin2 = np.empty((resWin))
    empWin2[:] = np.nan

    # empWinMax = np.empty((varMaxLen))
    # empWinMax[:] = np.nan

    trendPred = np.hstack((empWin, trTrain))

    trendPred = np.hstack((trendPred, empWin))
    trendPred = np.hstack((trendPred, trTest))

    resPred = np.hstack((empWin2, resTrain))
    resPred = np.hstack((resPred, empWin2))
    resPred = np.hstack((resPred, resTest))

    return trendPred,resPred

def plot(trainPred,trainY,testPred,testY):
    pred = np.concatenate((trainPred,testPred))
    gtruth = np.concatenate((trainY,testY))
    plt.plot(pred,'g')
    plt.plot(gtruth,'r')
    plt.show()

def LBtest(data):
    # lb,p = statsmodels.stats.diagnostic.acorr_ljungbox(residual)
    # print p
    r, q, p = sm.tsa.acf(data, qstat=True)
    data1 = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data1, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))

if __name__ == "__main__":
    #ts, data = load_data("./data/AEMO/NSW/nsw.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    ts, data = load_data("./data/AEMO/NSW/TAS2016.csv", indexName="SETTLEMENTDATE", columnName="TOTALDEMAND")
    #ts, data = load_data("./data/AEMO/TT30GEN.csv", indexName="TRADING_INTERVAL", columnName="VALUE")
    #ts, data = load_data("./data/bike/hour.csv", indexName="dteday", columnName="registered")
    #ts,data = load_data_xls("./data/air_quality/AirQuality.xlsx", indexName="Date", columnName="PT08.S2(NMHC)")
    #ts,data = load_data_txt("./data/house_data/house_power.csv", indexName="Date", columnName="Global_active_power")
    #ts, data = load_data_txt("./data/house_data/house_power.csv", indexName="Date", columnName="Global_active_power")#
    #data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    # print type(ts)
    #print data.shape
    #print (type(data[0]))

    plt.plot(data)
    plt.show()
    train,test = divideTrainTest(data,0.75)
    print (train.shape)
    print (test.shape)
    #trainX,trainY = createSamples(train,20, RNN=True)
    testX, testY = createSamples(test,20, RNN=True)
    #vtrainX,vtrainY = createVariableDataset(train, 10, 20,step=5)
    vtestX, vtestY = createVariableDataset(test, 10, 20, step=5)
    vtestY = transformGroundTruth(vtestY,10, 20, 5)
    #ptrainX, ptrainY = createPaddedDataset(train, 30, 40)
    #print (trainX.shape)
    #print (trainY.shape)
    print (testX.shape)
    print (testY.shape)
    #print (vtrainX.shape)
    #print (vtrainY.shape)
    print (vtestX.shape)
    print (vtestY.shape)
    # print (ptrainX.shape)
    # print (ptrainY.shape)






