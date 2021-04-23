import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from typing import *
from pathlib import Path
from enum import IntEnum
import pandas as pd

N_EPOCHS = 100
stock_price = 324.0
learning_rate = 0.01
r = 5 # hyperparameter,which has a better effect when equals to 5
df = pd.read_excel('training_data.xlsx')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fcn = torch.nn.MSELoss().to(device)
N_iterations = 70

def sig(x):
    return 1/1-np.exp(-x)


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class NaiveLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # Define/initialize all tensors
        # forget gate
        self.Wf = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bf = Parameter(torch.Tensor(hidden_sz))
        # input gate
        self.Wi = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bi = Parameter(torch.Tensor(hidden_sz))
        # Candidate memory cell
        self.Wc = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bc = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.Wo = Parameter(torch.Tensor(input_sz + hidden_sz, hidden_sz))
        self.bo = Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop bellow
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            hx_concat = torch.cat((ht, xt), dim=1)

            ### The LSTM Cell!
            ft = torch.sigmoid(hx_concat @ self.Wf + self.bf)
            it = torch.sigmoid(hx_concat @ self.Wi + self.bi)
            Ct_candidate = torch.tanh(hx_concat @ self.Wc + self.bc)
            ot = torch.sigmoid(hx_concat @ self.Wo + self.bo)
            # outputs
            Ct = ft * Ct + it * Ct_candidate
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)


class MogLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, mog_iterations: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.mog_iterations = mog_iterations
        # Define/initialize all tensors
        self.Wih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, input_sz))
        self.R = Parameter(torch.Tensor(input_sz, hidden_sz))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht):
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        # ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt, ht)  # mogrification
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)


# note that our hidden_sz is also our defined output size for each LSTM cell.
batch_sz, seq_len, feat_sz, hidden_sz = 1, 1, 1, 1
lstm2 = MogLSTM(feat_sz, hidden_sz, 5).to(device)
lstm1 = NaiveLSTM(feat_sz, hidden_sz).to(device)

optimizer1 = optim.SGD(lstm1.parameters(), lr=learning_rate)
optimizer2 = optim.SGD(lstm2.parameters(), lr=learning_rate)

plt.figure(figsize=(30,10))

for i in range(1, N_iterations):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    # Select two columns simulated data to train
    stock1 = round(df.iloc[i-1, 0])
    stock2 = round(df.iloc[i-1, 1])
    stock3 = round(df.iloc[i-1, 2])
    input = torch.tensor([[[stock1/stock_price], [stock2/stock_price]]]).to(device)

    output1, (hn, cn) = lstm1(input)
    output2, (hn, cn) = lstm2(input)
    output1 = output1.float()
    output2 = output2.float()
    y_t = torch.tensor([[[sig(stock3/stock_price)], [sig(stock3/stock_price)]]]).to(device)
    y_t1 = torch.zeros([1,2,1]).to(device)
    y_t = y_t.float()

    loss1 = loss_fcn(output1, y_t)
    loss2 = loss_fcn(output2, y_t)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()

    plt.subplot(1,3,1)
    plt.title('Naive LSTM')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.scatter(i, loss1.item())

    plt.subplot(1,3,2)
    plt.title('MOGRIFIER LSTM')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.scatter(i, loss2.item())

    plt.subplot(1,3,3)
    plt.title('Comparison')
    plt.xlabel('N_iterations')
    plt.ylabel('loss2/loss1')
    plt.scatter(i, loss2.item()/loss1.item())

plt.show()