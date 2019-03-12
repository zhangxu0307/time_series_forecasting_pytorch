'''
time series data loader for neural network models
'''
from torch.utils.data import Dataset
from src.util import *
import torch


class Time_Series_Data(Dataset):

    def __init__(self, train_x, train_y):
        self.X = train_x
        self.y = train_y

    def __getitem__(self, item):
        x_t = self.X[item]
        y_t = self.y[item]
        return x_t, y_t

    def __len__(self):

        return len(self.X)


if __name__ == '__main__':

    # ts, data = util.load_data("../data/NSW2013.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/bike_hour.csv", columnName="cnt")
    # ts, data = util.load_data("../data/TAS2016.csv", columnName="TOTALDEMAND")
    # ts, data = util.load_data("../data/traffic_data_in_bits.csv", columnName="value")
    # ts, data = util.load_data("../data/beijing_pm25.csv", columnName="pm2.5")
    ts, data = load_data("../data/pollution.csv", columnName="Ozone")

    trainData, testData = divideTrainTest(data)
    lag = 24
    flag = False
    trainX, trainY = createSamples(trainData, lag, RNN=flag)
    testX, testY = createSamples(testData, lag, RNN=flag)
    print("testX shape:", testX.shape)
    print("testy shape:", testY.shape)
    print("trainX shape:", trainX.shape)
    print("trainy shape:", trainY.shape)

    dataset = Time_Series_Data(trainX, trainY)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, sampler=None, batch_sampler=None, num_workers=4)

    for data, label in dataloader:
        print(data.shape)
        # print(data)
        print(label.shape)
        # print(label)
