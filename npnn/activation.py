import numpy as np


class Activation:
    def __init__(self):
        pass

    @staticmethod
    def calculate(x):
        return x

    @staticmethod
    def derivative(x):
        return x / x


class Linear(Activation):
    pass


class Relu(Activation):
    @staticmethod
    def calculate(x):
        return x.clip(0)

    @staticmethod
    def derivative(x):
        return x >= 0


class ReluFunc(Activation):
    @staticmethod
    def calculate(x):
        return 0.5 * (x + (x ** 2.0) ** 0.5)

    @staticmethod
    def derivative(x):
        return 0.5 + (0.5 * x) / ((x ** 2.0) ** 0.5)
