import pickle as p
import time
from src.model import *
from src.ts_loader import Time_Series_Data
# torch.manual_seed(1)
import numpy as np


def train(trainX, trainY,  lookBack, lr, modelPath, method, use_cuda=False,
          hidden_num=64, epoch=20, batchSize=32, checkPoint=10):

    lossFilePath = "../models/loss_ResRNN-4.pkl"
    output = open(lossFilePath, 'wb')
    lossList = []

    dataset = Time_Series_Data(trainX, trainY)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, sampler=None,
                                             batch_sampler=None, num_workers=1)
    net = None
    if method == "RNN":
        net = RNNModel(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=1, cell="RNN", use_cuda=use_cuda)
    if method == "LSTM":
        net = LSTMModel(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=1, cell="LSTM", use_cuda=use_cuda)
    if method == "GRU":
        net = GRUModel(inputDim=1, hiddenNum=hidden_num, outputDim=1, layerNum=1, cell="GRU", use_cuda=use_cuda)
    if method == "ResRNN":
        net = ResRNNModel(inputDim=1, hiddenNum=hidden_num, outputDim=1, resDepth=1, use_cuda=use_cuda)
    if method == "attention":
        net = RNN_Attention(inputDim=1, hiddenNum=hidden_num, outputDim=1, resDepth=4,
                            seq_len=lookBack, merge="concate", use_cuda=use_cuda)
    if method == "MLP":
        net = MLPModel(inputDim=lookBack, hiddenNum=hidden_num, outputDim=1)
    if use_cuda:
        net = net.cuda()
    net = net.train()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()

    t1 = time.time()
    lossSum = 0

    print("data loader num:", len(dataloader))

    for i in range(epoch):

        for batch_idx, (x, y) in enumerate(dataloader):

            x, y = Variable(x), Variable(y)
            if use_cuda:
                x = x.cuda()
                y = y.cuda()

            optimizer.zero_grad()

            pred = net.forward(x)
            loss = criterion(pred, y)

            lossSum += loss.item()
            if batch_idx % checkPoint == 0 and batch_idx != 0:
               print("batch: %d , loss is:%f" % (batch_idx, lossSum / checkPoint))
               lossList.append(lossSum / checkPoint)
               lossSum = 0

            loss.backward()
            optimizer.step()

        print("%d epoch is finished!" % i)

    t2 = time.time()
    print("train time:", t2-t1)
    p.dump(lossList, output, -1)

    torch.save(net, modelPath)

    return net


def predict(testX, net, use_cuda=False):

    if use_cuda:
        net = net.cuda()
    net = net.eval()
    testX = torch.from_numpy(testX)
    testX = Variable(testX)
    if use_cuda:
        testX = testX.cuda()
    pred = net(testX)
    if use_cuda:
        pred = pred.cpu()
    return pred.data.numpy()


def predict_iteration(net, testX, lookAhead, RNN=True, use_cuda=True):

    testBatchSize = testX.shape[0]
    ans = []

    for i in range(lookAhead):

        testX_torch = torch.from_numpy(testX)
        testX_torch = Variable(testX_torch)
        if use_cuda:
            testX_torch = testX_torch.cuda()
        pred = net(testX_torch)
        if use_cuda:
            pred = pred.cpu().data.numpy()
        else:
            pred = pred.data.numpy()
        pred = np.squeeze(pred)
        ans.append(pred)

        testX = testX[:, 1:]  # drop the head
        if RNN:
            pred = pred.reshape((testBatchSize, 1, 1))
            testX = np.append(testX, pred, axis=1)  # add the prediction to the tail
        else:
            pred = pred.reshape((testBatchSize, 1))
            testX = np.append(testX, pred, axis=1)  # add the prediction to the tail


    ans = np.array(ans)
    ans = ans.transpose([1, 0])
    return ans

