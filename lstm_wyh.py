import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.datautil import PBL_Dataset
from torch import optim

batch_size = 20
n_epoch = 20000
hidden_size = 50
num_layers = 1
num_days = 50
trade_limit = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Portfolio_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_days, num_layers, trade_limit):
        super(Portfolio_LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size = 1,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.total_days = input_size
        self.num_days = num_days
        self.trade_limit = trade_limit

        self.premium = nn.Linear(in_features = hidden_size, out_features = 1)
        self.act = nn.ReLU()
        self.trade_amount = nn.Linear(in_features = hidden_size, out_features = num_days)
        self.trade_days = nn.Linear(in_features = hidden_size, out_features = input_size - 1)
    
    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        last_output = hn[-1,:,:]

        premium = self.premium(last_output)
        premium = self.act(premium)

        trade_amount = self.trade_amount(last_output)
        trade_amount = torch.clamp(trade_amount, -self.trade_limit, self.trade_limit)

        trade_days = self.trade_days(last_output)
        trade_days = torch.sigmoid(trade_days)
        mx, days = torch.topk(trade_days, self.num_days)
        trade_days, _ = torch.sort(days, dim = 1)
        padding = torch.full((x.shape[0], 1), self.total_days - 1, device = device)
        trade_days = torch.cat([trade_days, padding], 1)

        return (premium, trade_amount, trade_days)

dataset = PBL_Dataset('./data/simulation-6.csv', repeat = 1)
train_dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(dataset = dataset, batch_size = 100, shuffle = False)

input_size = dataset.Days()

naive_lstm = Portfolio_LSTM(
    input_size = input_size,
    hidden_size = hidden_size,
    num_days = num_days,
    num_layers = num_layers,
    trade_limit = trade_limit
    ).to(device)

mse_loss = nn.MSELoss(reduction = 'mean').to(device)

def Gst(st):
    return st.to(device)

def Cost(input, premium, Delta, Ts):
    XT = []
    for i in range(input.shape[0]):
        st = input[i]
        p = premium[i]
        delta = Delta[i]
        ts = Ts[i]
        tq = torch.index_select(st, 0, ts[1:]) - torch.index_select(st, 0, ts[:-1])
        xt = torch.dot(delta, tq.reshape(-1)) + p
        XT.append(xt)
    return torch.tensor(XT, requires_grad = True).to(device)

def train(model, loss_fcn):
    adam = optim.SGD(model.parameters(), lr = 1e-3)
    model.train()

    for epoch in range(n_epoch):
        losses = []
        for batch_x in train_dataloader:
            input = batch_x.to(device)

            adam.zero_grad()
            p, delta, ts = model(input)

            xt = Cost(input, p, delta, ts)
            gst = Gst(input[:,-1,:].reshape(-1))
            loss = loss_fcn(xt, gst)

            losses.append(loss.item())
            loss.backward()
            adam.step()

        print(np.mean(losses))
        

train(naive_lstm, mse_loss)