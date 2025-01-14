import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]

def forward(x):
    return w * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2  # 修正为平方形式

# 存储w、b和对应的MSE
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 2.1, 0.5):
        print('w=', w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        mse = l_sum / 3
        print('MSE=', mse)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(w_list, b_list, mse_list)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()