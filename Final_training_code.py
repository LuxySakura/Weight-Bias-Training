import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from typing import *
from enum import IntEnum
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


learning_rate = 2
r = 4  # hyperparameter,which has a better effect when equals to 5
delta = 0.5
batch_size = 50
n_epoch = 1000
hidden_size = 50
num_layers = 1
num_days = 50
trade_limit = 10


class PBL_Dataset(Dataset):
    def __init__(self, filename, repeat=1):
        dat = np.loadtxt(filename, delimiter=',', dtype=float, encoding='utf-8-sig')
        dat /= dat[0]
        dat = dat.T
        dat = dat.reshape(dat.shape[0], dat.shape[1], 1)
        self.data = torch.Tensor(dat)
        self.len = self.data.shape[0]
        self.repeat = repeat

    def __getitem__(self, index):
        idx = index % self.len
        return self.data[idx]

    def __len__(self):
        if self.repeat == None:
            return 100000000
        return self.len * self.repeat

    def Days(self):
        return self.data.shape[1]


dataset = PBL_Dataset('./data/simulation-new.csv', repeat=1)
neutral_dataset = PBL_Dataset('./data/simulation-neutral.csv', repeat=1)


class Asy_loss_function(nn.Module):
    def __init__(self, learning_rate):
        super(Asy_loss_function, self).__init__()
        self.learning_rate = learning_rate
        return

    def forward(self, output, target):
        y = torch.mean(output-target)
        if y >= 0:
            return (1+self.learning_rate)*torch.mean(torch.pow((output - target), 2))
        elif y < 0:
            return torch.mean(torch.pow((output - target), 2))


class Huber_loss_function(nn.Module):
    def __init__(self, delta):
        super(Huber_loss_function, self).__init__()
        self.delta = delta
        return

    def forward(self, output, target):
        y = torch.mean(output-target)
        if y <= self.delta:
            return 0.5*torch.mean(torch.pow((output - target), 2))
        elif y > self.delta:
            return self.delta*y - 0.5*self.delta*self.delta


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


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
                ht = (2 * torch.sigmoid(torch.mm(xt, self.R))) * ht
            else:
                xt = (2 * torch.sigmoid(torch.mm(ht, self.Q))) * xt
        return xt, ht

    # Define forward pass through all LSTM cells across all timesteps.
    # By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = [0] * seq_sz
        # ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt, ht)  # mogrification
            gates = (torch.mm(xt, self.Wih) + self.bih) + (torch.mm(ht, self.Whh) + self.bhh)
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

            hidden_seq[t] = ht.unsqueeze(Dim.batch)
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (ht, Ct)


class Portfolio_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_days, trade_limit):
        super(Portfolio_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.total_days = input_size
        self.num_days = num_days
        self.trade_limit = trade_limit

        self.premium = nn.Linear(in_features=hidden_size, out_features=1)
        self.act = nn.ReLU()
        # self.trade_amount = nn.Linear(in_features = hidden_size, out_features = num_days)
        self.trade_amount = nn.Linear(in_features=hidden_size, out_features=input_size - 1)
        # self.trade_days = nn.Linear(in_features = hidden_size, out_features = input_size - 1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        last_output = hn[-1, :, :]

        premium = self.premium(last_output)
        premium = self.act(premium)

        trade_amount = self.trade_amount(last_output)
        trade_amount = torch.clamp(trade_amount, -self.trade_limit, self.trade_limit)

        # trade_days = self.trade_days(last_output)
        # trade_days = torch.sigmoid(trade_days)
        # mx, days = torch.topk(trade_days, self.num_days)
        # trade_days, _ = torch.sort(days, dim = 1)
        # padding = torch.full((x.shape[0], 1), self.total_days - 1, device = device)
        # trade_days = torch.cat([trade_days, padding], 1)

        # return (premium, trade_amount, trade_days)
        return (premium, trade_amount)


class Portfolio_Mog(nn.Module):
    def __init__(self, input_size, hidden_size, num_days, trade_limit, mog_iterations):
        super(Portfolio_Mog, self).__init__()

        self.lstm = MogLSTM(input_sz=input_size, hidden_sz=hidden_size, mog_iterations=mog_iterations)

        self.total_days = input_size
        self.num_days = num_days
        self.trade_limit = trade_limit

        self.premium = nn.Linear(in_features=hidden_size, out_features=1)
        self.act = nn.ReLU()
        # self.trade_amount = nn.Linear(in_features = hidden_size, out_features = num_days)
        self.trade_amount = nn.Linear(in_features=hidden_size, out_features=input_size - 1)
        # self.trade_days = nn.Linear(in_features = hidden_size, out_features = input_size - 1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        last_output = hn[-1, :]

        premium = self.premium(last_output)
        premium = self.act(premium)

        trade_amount = self.trade_amount(last_output)
        trade_amount = torch.clamp(trade_amount, -self.trade_limit, self.trade_limit)

        return (premium, trade_amount)


class Portfolio_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_days, num_layers, trade_limit):
        super(Portfolio_RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.total_days = input_size
        self.num_days = num_days
        self.trade_limit = trade_limit

        self.premium = nn.Linear(in_features=hidden_size, out_features=1)
        self.act = nn.ReLU()
        # self.trade_amount = nn.Linear(in_features = hidden_size, out_features = num_days)
        self.trade_amount = nn.Linear(in_features=hidden_size, out_features=input_size - 1)
        # self.trade_days = nn.Linear(in_features = hidden_size, out_features = input_size - 1)

    def forward(self, x):
        output, hn = self.rnn(x)
        last_output = hn[-1, :, :]

        premium = self.premium(last_output)
        premium = self.act(premium)

        trade_amount = self.trade_amount(last_output)
        trade_amount = torch.clamp(trade_amount, -self.trade_limit, self.trade_limit)

        # trade_days = self.trade_days(last_output)
        # trade_days = torch.sigmoid(trade_days)
        # mx, days = torch.topk(trade_days, self.num_days)
        # trade_days, _ = torch.sort(days, dim = 1)
        # padding = torch.full((x.shape[0], 1), self.total_days - 1, device = device)
        # trade_days = torch.cat([trade_days, padding], 1)

        # return (premium, trade_amount, trade_days)
        return (premium, trade_amount)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Asy_loss = Asy_loss_function(learning_rate)
MSE_loss = torch.nn.MSELoss()
Huber_loss = Huber_loss_function(delta)


def Gst(st):
    return st.to(device)


def Cost(input, premium, Delta):
    st = input.squeeze()
    ds = st[:, 1:] - st[:, :-1]
    cost = torch.sum(ds * Delta, dim=1)
    xt = cost + premium.squeeze()
    return xt


def train(model, loss_fcn, dataset):
    slen = len(dataset)
    train_size, test_size = int(0.8*slen), int(0.2*slen)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=10, shuffle=True)

    adam = optim.Adam(model.parameters(), lr=1e-3)

    train_loss = []
    test_loss = []

    for epoch in trange(n_epoch):
        model.train()
        losses = []
        for batch_x in train_dataloader:
            input = batch_x.to(device)

            adam.zero_grad()
            # p, delta, ts = model(input)
            p, delta = model(input)

            # xt = Cost(input, p, delta, ts)
            xt = Cost(input, p, delta)
            gst = Gst(input[:, -1, :].reshape(-1))
            loss = loss_fcn(xt, gst)
            losses.append(loss.item())
            loss.backward()
            adam.step()
        train_loss.append(np.mean(losses))

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_x in test_dataloader:
                input = batch_x.to(device)
                p, delta = model(input)
                xt = Cost(input, p, delta)
                gst = Gst(input[:, -1, :].reshape(-1))
                loss = loss_fcn(xt, gst)
                losses.append(loss.item())
            test_loss.append(np.mean(losses))
    return train_loss, test_loss


input_size = dataset.Days()
LSTM = Portfolio_LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_days=num_days, trade_limit=trade_limit).to(device)
Mog_LSTM = Portfolio_Mog(input_size=input_size, hidden_size=hidden_size,
                         num_days=num_days, trade_limit=trade_limit, mog_iterations=r).to(device)
RNN = Portfolio_RNN(input_size=input_size, hidden_size=hidden_size,
                    num_days=num_days, num_layers=num_layers, trade_limit=trade_limit).to(device)

LSTM2 = Portfolio_LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_days=num_days, trade_limit=trade_limit).to(device)
Mog_LSTM2 = Portfolio_Mog(input_size=input_size, hidden_size=hidden_size,
                         num_days=num_days, trade_limit=trade_limit, mog_iterations=r).to(device)
RNN2 = Portfolio_RNN(input_size=input_size, hidden_size=hidden_size,
                    num_days=num_days, num_layers=num_layers, trade_limit=trade_limit).to(device)

LSTM3 = Portfolio_LSTM(input_size=input_size, hidden_size=hidden_size,
                      num_days=num_days, trade_limit=trade_limit).to(device)
Mog_LSTM3 = Portfolio_Mog(input_size=input_size, hidden_size=hidden_size,
                         num_days=num_days, trade_limit=trade_limit, mog_iterations=r).to(device)
RNN3 = Portfolio_RNN(input_size=input_size, hidden_size=hidden_size,
                    num_days=num_days, num_layers=num_layers, trade_limit=trade_limit).to(device)

loss_table = pd.DataFrame()

def work_table(model, loss_fcn, model_name, loss_name):
    global loss_table
    train_loss, test_loss = train(model, loss_fcn, dataset)
    table = pd.DataFrame({"epoch": np.arange(1, n_epoch+1), "model": model_name, "loss_fcn": loss_name, 
                            "train_loss": train_loss, "test_loss": test_loss})
    loss_table = loss_table.append(table, ignore_index = True)

#work_table(LSTM, Huber_loss, "LSTM", "Huber")
work_table(RNN, Huber_loss, "RNN", "Huber")
#work_table(Mog_LSTM, Huber_loss, "Mog", "Huber")

#work_table(LSTM2, Asy_loss, "LSTM", "Asym")
work_table(RNN2, Asy_loss, "RNN", "Asym")
#work_table(Mog_LSTM2, Asy_loss, "Mog", "Asym")

#work_table(LSTM3, MSE_loss, "LSTM", "MSE")
work_table(RNN3, MSE_loss, "RNN", "MSE")
#work_table(Mog_LSTM3, MSE_loss, "Mog", "MSE")

loss_table.to_csv("loss_data_RNN.csv", index = False)
