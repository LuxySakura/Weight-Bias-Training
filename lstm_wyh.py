import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.datautil import PBL_Dataset

batch_size = 50
n_epoch = 20000
hidden_size = 50
num_layers = 1
num_days = 50
trade_limit = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Portfolio_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_days, num_layers, trade_limit):
        super(Portfolio_LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.num_days = num_days
        self.trade_limit = trade_limit

        self.premium = nn.Linear(in_features = hidden_size, out_features = 1)
        self.trade_amount = nn.Linear(in_features = hidden_size, out_features = num_days)
        self.trade_days = nn.Linear(in_features = hidden_size, out_features = input_size - 1)
    
    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        last_output = hn[-1,:,:]

        premium = self.premium(last_output)

        trade_amount = self.trade_amount(last_output)
        trade_amount = torch.clamp(trade_amount, -self.trade_limit, self.trade_limit)

        trade_days = self.trade_days(last_output)
        trade_days = torch.sigmoid(trade_days)

        return (premium, trade_amount, trade_days)

dataset = PBL_Dataset('./data/simulation-6.csv', repeat = None)
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
