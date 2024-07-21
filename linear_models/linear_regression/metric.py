import numpy as np


class Metric:
    def __init__(self, name):
        self.name = name
        self._function = getattr(self, name)

    def calculate(self, y_fact, y_predicted):
        return self._function(y_fact, y_predicted)

    @classmethod
    def mae(cls, y_fact, y_predicted):
        return np.mean(np.abs(y_fact - y_predicted))

    @classmethod
    def mse(cls, y_fact, y_predicted):
        return np.mean((y_fact - y_predicted) ** 2)

    @classmethod
    def rmse(cls, y_fact, y_predicted):
        return np.sqrt(cls.mse(y_fact, y_predicted))

    @classmethod
    def r2(cls, y_fact, y_predicted):
        y_mean = np.mean(y_fact)
        return 1 - np.sum((y_fact - y_predicted) ** 2) / np.sum((y - y_mean) ** 2)

    @classmethod
    def mape(cls, y_fact, y_predicted):
        return 100 * np.mean(np.abs((y_fact - y_predicted) / y_fact))
