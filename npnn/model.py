from npnn.layer import Dense
from npnn.activation import Relu, Linear
from npnn.loss import MAE, MSE
import numpy as np

lr = 0.00001
l1 = Dense(5, 12, Relu, lr=lr)
l2 = Dense(12, 12, Relu, lr=lr)
l3 = Dense(12, 2, Linear, lr=lr)
losses = []

iterations = 1000000
for i in range(0, iterations):

    x = np.random.randint(low=1, high=5, size=(64 * 4, 5))
    y = x[:, 1:3]
    y[:, 1] = x[:, 2]**2 - 5.0

    y1 = l1.forward(x)
    y2 = l2.forward(y1)
    y3 = l3.forward(y2)

    e = MSE.calculate(y3, y)
    e_dx = MSE.derivative(y3, y)
    e3 = e * e_dx
    # e3 = (y2 - y)
    # e = e3

    e2 = l3.backward(y2, y3, e3)
    e1 = l2.backward(y1, y2, e2)
    e0 = l1.backward(x, y1, e1)

    loss = np.sum(e)
    losses.append(loss)

    if i % 10000 == 0:  # LR Decay ExponentialDecay
        l1.lr *= 1.0  # 0.98
        l2.lr *= 1.0  # 0.98
        l3.lr *= 1.0  # 0.98

    if i % 50000 == 0:
        print(f'i:{i} ({np.round(100 * i / iterations, 2)} %) loss:{np.round(loss, 10)}    lr: {l1.lr}')

print(np.round(l1.w, 2))
print(np.round(l2.w, 2))
print(np.round(l3.w, 2))
print()

print(y[0:5])
print(y3[0:5])

#####
import matplotlib.pyplot as plt

plt.plot(losses)
plt.yscale('log')
plt.show()
