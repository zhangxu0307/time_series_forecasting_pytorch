import pickle as p

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.pyplot import plot, savefig

pkl_file1 = open('../models/loss_RNN.pkl', 'rb')
pkl_file2 = open('../models/loss_GRU.pkl', 'rb')
pkl_file3 = open('../models/loss_LSTM.pkl', 'rb')
pkl_file4 = open('../models/loss_ResRNN-2.pkl', 'rb')
pkl_file5 = open('../models/loss_ResRNN-4.pkl', 'rb')
pkl_file6 = open('../models/loss_ResRNN-0.pkl', 'rb')


RNN = p.load(pkl_file1)
GRU = p.load(pkl_file2)
LSTM = p.load(pkl_file3)
RES2 = p.load(pkl_file4)
RES4 = p.load(pkl_file5)
RES0 = p.load(pkl_file6)


plt.plot(RNN, label="RNN", c="r", marker='^')
plt.plot(GRU, label="GRU", c="g", marker='o')
plt.plot(LSTM, label="LSTM",marker='+')
plt.plot(RES2, label="TSR-RNN-2", marker='x')
plt.plot(RES4, label="TSR-RNN-4", marker='*')
#plt.plot(RES0, label="RES0")
plt.legend()
savefig('../models/MyFig2.eps')
#plt.show()
