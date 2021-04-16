import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# 随机初始化数据
beta1 = 0.6
beta2 = 0.8
N_iterations = 20
N_batch = 2
learning_rate = 0.01
t = 3
stock_price = 119.0  # g(T)

m_w0 = torch.zeros([8, 2])  # store weight0
m_w1 = torch.zeros([8, 2])  # store weight1
m_b0 = torch.zeros([1, 8])  # store bias0
m_b1 = torch.zeros([1, 8])  # store bias1

v_w0 = torch.zeros([8, 2])  # store weight0
v_w1 = torch.zeros([8, 2])  # store weight1
v_b0 = torch.zeros([1, 8])  # store bias0
v_b1 = torch.zeros([1, 8])  # store bias1

theta_w0 = torch.randn([8, 2])  # store weight0
theta_w1 = torch.randn([8, 2])  # store weight1
theta_b0 = torch.randn([1, 8])  # store bias0
theta_b1 = torch.randn([1, 8])  # store bias1

# 读取表格信息
df = pd.read_excel('training_data.xlsx')

# 构建神经网络
net2 = torch.nn.LSTM(2, 2, 1)
input = torch.randn(1, 1, 2)


loss_fcn = torch.nn.MSELoss()

y_t = torch.zeros([1, 1, 2])




for i in range(1, N_iterations ):

    # 选取两列不同年分的股票数据投入网络中进行训练
    stock1 = float(round(df.iloc[i-1, 0]))
    stock2 = float(round(df.iloc[i-1, 1]))
    stock3 = round(df.iloc[i-1, 2])
    input= torch.sigmoid(torch.tensor([[[stock1, stock2]]]))
    h0 = torch.zeros([1, 1, 2])
    c0 = torch.zeros([1, 1, 2])
    output, (hn, cn) = net2(input, (h0, c0))
    y_t = torch.zeros([1, 1, 2])
    loss = loss_fcn(output, y_t)
    loss.backward()
    n = 0
    for param in net2.named_parameters():
        if n == 0:
            g_w0 = param[1].grad
            n = n + 1
        elif n == 1:
            g_w1 = param[1].grad
            n = n + 1
        elif n == 2:
            g_b0 = param[1].grad
            n = n + 1
        else:
            g_b1 = param[1].grad

    m_w0 = m_w0 + (1 - beta1) * g_w0
    m_b0 = m_b0 + (1 - beta1) * g_b0
    m_w1 = m_w1 + (1 - beta1) * g_w1
    m_b1 = m_b1 + (1 - beta1) * g_b1
    v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
    v_w1 = beta2 * v_w1 + (1 - beta2) * g_w1 * g_w1
    v_b0 = beta2 * v_b0 + (1 - beta2) * g_b0 * g_b0
    v_b1 = beta2 * v_b1 + (1 - beta2) * g_b1 * g_b1

    m_hatw0 = m_w0 / (1 - pow(beta1, t))
    m_hatw1 = m_w1 / (1 - pow(beta1, t))
    m_hatb0 = m_b0 / (1 - pow(beta1, t))
    m_hatb1 = m_b1 / (1 - pow(beta1, t))
    v_hatw0 = beta2 * v_w0 / (1 - pow(beta2, t))
    v_hatw1 = beta2 * v_w1 / (1 - pow(beta2, t))
    v_hatb0 = v_b0 / (1 - pow(beta2, t))
    v_hatb1 = v_b1 / (1 - pow(beta2, t))

    theta_w0 = theta_w0 - learning_rate * m_hatw0 / np.sqrt(v_hatw0)
    theta_w1 = theta_w0 - learning_rate * m_hatw1 / np.sqrt(v_hatw1)
    theta_b0 = theta_b0 - learning_rate * m_hatb0 / np.sqrt(v_hatb0)
    theta_b0 = theta_b0 - learning_rate * m_hatb1 / np.sqrt(v_hatb1)
    t = t + 1

print('#weight 0', theta_w0)
print('#weight 1', theta_w1)
print('#bias 0', theta_b0)
print('#bias 1', theta_b1)
print(stock2)

