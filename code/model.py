
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

# 模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = layerNum
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)


# 标准RNN模型
class  RNNModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):

        super(RNNModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    # def init_hidden(self, batchSize):
    #
    #     return hidden

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0) # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


# LSTM模型
class LSTMModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(LSTMModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)


    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, (h0, c0))  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn[0].view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput

# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, layerNum, cell):
        super(GRUModel, self).__init__(inputDim, hiddenNum, outputDim, layerNum, cell)

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        rnnOutput, hn = self.cell(x, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput

# ResRNN模型
class ResRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth):

        super(ResRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        #self.tanh = nn.Tanh()

    def forward(self, x, batchSize):

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        # output = []
        inputLen = x.data.size()[1]
        ht = h0
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            if self.resDepth == 0:
                h0 = nn.Tanh()(hn)
            if self.resDepth == 1:
                # res depth = 1
                h0 = nn.Tanh()(hn + h0)
            if self.resDepth >= 2:
                # res depth = N
                if i % self.resDepth == 0 and i != 0:
                    h0 = nn.Tanh()(hn + ht)
                    ht = hn
                else:
                    h0 = nn.Tanh()(hn)


            # 首尾加入res
            if self.resDepth == -1:
                if i == 0:
                    hstart = hn
                if i == inputLen-2:
                    h0 = nn.Tanh()(hn+hstart)
                else:
                    if i % 4 == 0 and i != 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)

        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput

# 加入注意机制的RNN模型
class AttentionRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, seqLen):
        super(AttentionRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum*seqLen, self.outputDim, bias=True)
        # self.tanh = nn.Tanh()

    def forward(self, x, batchSize):
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        hiddenList = []
        inputLen = x.data.size()[1]
        for i in range(inputLen):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)
            h0 = nn.Tanh()(hn)
            ht = h0.view(batchSize, self.hiddenNum)
            hiddenList.append(ht)
        flanten = torch.cat(hiddenList, dim=1)

        fcOutput = self.fc(flanten)

        return fcOutput


# 标准ANN模型
class ANNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim):

        super(ANNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.fc1 = nn.Linear(self.inputDim, self.hiddenNum)
        self.fc2 = nn.Linear(self.hiddenNum, self.outputDim)

    def forward(self,x, batchSize):

        output = self.fc1(x)
        output = self.fc2(output)

        return output

# 分解网络
class DecompositionNetModel(nn.Module):

    def __init__(self, inputDim, fchiddenNum, rnnhiddenNum, outputDim):

        super(DecompositionNetModel, self).__init__()
        self.fchiddenNum = fchiddenNum
        self.rnnhiddenNum = rnnhiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.rnnInputDim = 1

        # dropout层
        self.drop = nn.Dropout(p=0.3)

        # 一维卷积层
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2,  bias=True)
        self.pool = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        #self.conv.weight.data.fill_(0.2)
        self.convWeight = self.conv.weight.data
        #print(self.conv.weight.data)

        # 全连接层
        self.fc1 = nn.Linear(self.inputDim, self.fchiddenNum)
        self.fc2 = nn.Linear(self.fchiddenNum, self.inputDim)

        # 循环神经网络层
        self.rnn1 = nn.RNN(input_size=self.rnnInputDim, hidden_size=self.rnnhiddenNum,
                           num_layers=self.layerNum, dropout=0.5,
                           nonlinearity="tanh", batch_first=True, )
        self.rnn2 = nn.RNN(input_size=self.rnnInputDim, hidden_size=self.rnnhiddenNum,
                          num_layers=self.layerNum, dropout=0.5,
                          nonlinearity="tanh", batch_first=True, )
        self.resrnn1 = ResRNNModel(inputDim=1, hiddenNum=self.rnnhiddenNum, outputDim=1, resDepth=4)
        self.resrnn2 = ResRNNModel(inputDim=1, hiddenNum=self.rnnhiddenNum, outputDim=1, resDepth=4 )
        self.gru1 = nn.GRU(input_size=self.rnnInputDim, hidden_size=self.rnnhiddenNum,
                           num_layers=self.layerNum, dropout=0.0,
                           batch_first=True, )
        self.gru2 = nn.GRU(input_size=self.rnnInputDim, hidden_size=self.rnnhiddenNum,
                           num_layers=self.layerNum, dropout=0.0,
                           batch_first=True, )

        # 线性输出层
        self.fc3 = nn.Linear(self.rnnhiddenNum, self.outputDim)
        self.fc4 = nn.Linear(self.rnnhiddenNum, self.outputDim)

    def forward(self, x, batchSize):

        # 分解网络
        x = torch.unsqueeze(x, 1)
        #print(x.size())
        #x = torch.transpose(x, 1, 2)
        # output = self.fc1(x)
        # prime = self.fc2(output)
        prime = self.conv(x)
        #print(prime.size())
        prime = self.pool(prime)
        #print(prime.size())
        residual = x-prime
        # prime = torch.unsqueeze(prime, 2)
        # residual = torch.unsqueeze(residual, 2)
        prime = torch.transpose(prime, 1, 2)
        residual = torch.transpose(residual, 1, 2)

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.rnnhiddenNum))

        # 预测主成分rnn网络
        rnnOutput1, hn1 = self.gru1(prime, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn1 = hn1.view(batchSize, self.rnnhiddenNum)
        #hn1 = self.drop(hn1)
        fcOutput1 = self.fc3(hn1)
        #fcOutput1 = self.resrnn1.forward(prime, batchSize=batchSize)

        # 预测残差rnn网络
        rnnOutput2, hn2 = self.gru2(residual, h0)  # rnnOutput 12,20,50 hn 1,20,50
        hn2 = hn2.view(batchSize, self.rnnhiddenNum)
        #hn2 = self.drop(hn2)
        fcOutput2 = self.fc4(hn2)
        #fcOutput2 = self.resrnn2.forward(prime, batchSize=batchSize)

        # 合并预测结果
        result = fcOutput1+fcOutput2

        return result, fcOutput1, fcOutput2, residual



