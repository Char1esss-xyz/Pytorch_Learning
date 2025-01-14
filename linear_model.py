import numpy as np
import matplotlib.pyplot as plt

#穷举法预测线性模型

#训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#预测函数
def forward(x):
    return x * w

#计算损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

#创建w和Mean_Square_Error数组
w_list = []
mse_list = []

#从0到4穷举
for w in np.arange(0.0, 4.1, 0.1):
    print('w=',w)
    l_sum = 0
    #穷举w为具体值时的预测值和误差
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    #输出训练集的平均损失
    print('MSE=', l_sum / 3)
    #记录w和mse
    w_list.append(w)
    mse_list.append(l_sum / 3)
    
#输出图像
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
