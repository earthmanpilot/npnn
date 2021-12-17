import numpy as np


class MAE:
    def __init__(self):
        pass

    @staticmethod
    def calculate(p, y):
        e = np.abs(p - y)
        e = e / e.size
        return e

    @staticmethod
    def derivative(p, y):
        return np.sign(p-y)


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def calculate(p, y):
        e = np.square(p - y)
        e = e / e.size
        return e

    @staticmethod
    def derivative(p, y):
        return 2.*(p-y)
