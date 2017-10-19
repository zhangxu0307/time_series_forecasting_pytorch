from code.model import RNNModel, LSTMModel, GRUModel, ANNModel, ResRNNModel, AttentionRNNModel, DecompositionNetModel
from torch.autograd import Variable
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
import pickle as p
import time

def customLoss(pred, fcOutput1, fcOutput2, residual, convWeight, y):

    lamda1 = 0.01
    lamda2 = 2.0
    lamda3 = 0.1

    criterion = nn.MSELoss()
    loss = criterion(pred, y)
    #print(convWeight)
    #print(torch.norm(convWeight, 2))
    # lamda3*torch.norm(residual, 2)
    #lamda1 * criterion(fcOutput1, y) + lamda2 * criterion(fcOutput2, y)
    return loss

def train(trainX, trainY, epoch, lr, batchSize, modelPath, lookBack, method):

    lossFilePath = "../model/loss_ResRNN-4.pkl"
    output = open(lossFilePath, 'wb')
    lossList = []

    n = trainX.shape[0]
    print("trainx num is:", n)
    batchNum = n//batchSize-1

    print("batch num is:", batchNum)

    if method == "RNN":
        net = RNNModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="RNN")
    if method == "LSTM":
        net = LSTMModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="LSTM")
    if method == "GRU":
        net = GRUModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="GRU")
    if method == "ResRNN":
        #net = ResidualRNNModel(inputDim=1, hiddenNum=100, outputDim=1, layerNum=1, cell="RNNCell")
        net = ResRNNModel(inputDim=1, hiddenNum=100, outputDim=1, resDepth=-1)
    if method == "attention":
        net = AttentionRNNModel(inputDim=1, hiddenNum=100, outputDim=1, seqLen=lookBack)

    if method == "ANN":
        net = ANNModel(inputDim=lookBack, hiddenNum=100, outputDim=1)

    if method == "new":
        net = DecompositionNetModel(inputDim=lookBack, fchiddenNum=100, rnnhiddenNum=100, outputDim=1)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #optimizer = optim.SGD(net.parameters(), lr=0.001)


    t1 = time.time()
    for i in range(epoch):
        trainX, trainY = shuffle(trainX, trainY,
                                 random_state=epoch
                                 )
        batchStart = 0
        lossSum = 0

        for j in range(batchNum):

            x = trainX[batchStart:batchStart+batchSize, :, :]
            y = trainY[batchStart:batchStart+batchSize]

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            x, y = Variable(x), Variable(y)


            optimizer.zero_grad()

            if method == "new":
                pred, fcOutput1, fcOutput2, resdiual = net.forward(x, batchSize=batchSize)
                criterion = nn.MSELoss()
                #loss = criterion(pred, y)
                loss = customLoss(pred, fcOutput1, fcOutput2, resdiual, net.convWeight, y)
            else:
                pred = net.forward(x, batchSize=batchSize)
                criterion = nn.MSELoss()
                loss = criterion(pred, y)

            lossSum += loss.data.numpy()[0]
            if j % 30 == 0 and j != 0:
               print("current loss is:", lossSum/10)
               lossList.append(lossSum/10)
               lossSum = 0

            #net.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)

            batchStart += batchSize
        print("%d epoch is finished!" %i)
    t2 = time.time()
    print("train time:", t2-t1)
    p.dump(lossList, output, -1)

    torch.save(net, modelPath)


def predict(testX, modelFileName):

    net = torch.load(modelFileName)
    testBatchSize = testX.shape[0]
    testX = torch.from_numpy(testX)
    testX = Variable(testX)
    pred = net(testX, testBatchSize)

    return pred.data.numpy()

