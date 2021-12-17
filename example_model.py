from npnn.layer import Dense
from npnn.activation import Relu
import numpy as np

from datetime import datetime
s = datetime.now()

lr = 0.000005
l1 = Dense(5,100, Relu, lr=lr)
l2 = Dense(100,100, Relu, lr=lr)
l3 = Dense(100,2, Relu, lr=lr)
losses = []

iter = 1000000
for i in range(0, iter):

    x = np.random.randint(low=1, high=5, size=(64*4, 5))
    #x = np.random.rand(16,5)
    y = x[:, 1:3]
    #y[:,0] = x[:,1] + x[:,3]
    #y[:,1] = x[:,0]

    y1 = l1.forward(x)
    y2 = l2.forward(y1)
    y3 = l3.forward(y2)

    e3 = y3-y
    e2 = l3.backward(y2,y3, e3)
    e1 = l2.backward(y1,y2, e2)
    e0 = l1.backward(x,y1, e1)
    e = e3

    loss = np.mean(np.abs( e ))
    losses.append(loss)

    if i % 250000 == 0:
        print(f'i:{i} ({np.round(100*i/iter,2)} %) loss:{np.round(loss, 10)}')
print( np.round(l1.w,2))
print( np.round(l2.w,2))
print( np.round(l3.w,2))
print()

print(y[0:5])
print(y3[0:5])
print(datetime.now() - s)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.yscale('log')
plt.show()