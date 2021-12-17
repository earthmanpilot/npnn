import inspect
from npnn import activation
import numpy as np
import unittest
from unittest import TestCase


class ActicationTest(TestCase):
    def test_check_differential(self):
        delta_x = 0.0000001
        x = np.array([-10.0, -2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 10.0])
        for name, obj in inspect.getmembers(activation, inspect.isclass):
            dx_aprox = (obj.calculate(x + delta_x) - obj.calculate(x)) / delta_x
            dx = obj.derivative(x)

            assert np.all(
                np.round(dx_aprox.astype(float), 2) == np.round(dx.astype(float), 2)
            ), f"{name} failed derivative testing"


if __name__ == "__main__":
    unittest.main()
