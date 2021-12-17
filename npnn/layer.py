import numpy as np
from npnn.activation import Activation


class Layer:
    def __init__(self, in_size: int, out_size: int, activation: Activation):
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.x_w = None
        self.x_a = None

    def __repr__(self):
        return f"{self.in_size}, {self.out_size}, {self.w.shape}, {self.w.size}"


class Dense(Layer):
    def __init__(self, in_size: int, out_size: int, activation: Activation, lr=0.0001):
        super().__init__(in_size, out_size, activation)
        #self.w =  np.random.normal(0.0, 0.05, size=(in_size, out_size))
        self.w = np.random.uniform(-np.sqrt(6/(in_size+out_size)),
                                   np.sqrt(6/(in_size+out_size)),
                                   size=(in_size, out_size))
        self.lr = lr

    def forward(self, x: np.array):
        x_w = x.dot(self.w)
        x_a = self.activation.calculate(x_w)
        self.x_w = x_w
        self.x_a = x_a
        return x_a

    def backward(self, x: np.array, y: np.array, e):
        e_da = e * self.activation.derivative(y)
        e_dw = x.T.dot(e_da)
        self.w -= self.lr * e_dw
        up_e = e_da.dot(self.w.T)
        return up_e
