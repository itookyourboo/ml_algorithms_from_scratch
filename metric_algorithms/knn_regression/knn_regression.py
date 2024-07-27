import pandas as pd
import numpy as np

from metric_algorithms.knn_regression.weight import Weight
from metric_algorithms.metric import Metric


class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = (0, 0)
        self.metric = Metric(metric)
        self.weight = Weight(weight)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame):
        y = np.zeros(X.shape[0])
        for i, x1 in enumerate(X.values):
            knn = sorted(
                (self.metric.get(x1, x2), y2)
                for x2, y2 in zip(self.X.values, self.y.values)
            )[:self.k]
            y[i] = self.weight.get(knn)
        return y

    def __str__(self):
        return f'{self.__class__.__name__} class: k={self.k}'
