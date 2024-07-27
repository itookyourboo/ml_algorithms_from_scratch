import numpy as np
import pandas as pd

from metric_algorithms.metric import Metric
from metric_algorithms.knn_classification.weight import Weight


class MyKNNClf:
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
        return np.floor(self.predict_proba(X) + .5).astype(int)

    def predict_proba(self, X: pd.DataFrame):
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


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X_train, y_train = make_regression(n_samples=100, n_features=14, n_informative=10, noise=15, random_state=42)
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train % 1 < 0.5)
    X_train.columns = [f'col_{col}' for col in X_train.columns]

    knn_clf = MyKNNClf(k=2)
    knn_clf.fit(X_train, y_train)

    X_test, y_test = make_regression(n_samples=100, n_features=14, n_informative=10, noise=15, random_state=42)
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test % 1 < 0.5)
    X_test.columns = [f'col_{col}' for col in X_test.columns]

    predicted = knn_clf.predict(X_test)
    probas = knn_clf.predict_proba(X_test)
    print(np.rint([0.5, 8, 0.8, 0.2, 0.501]))
    print(predicted)
    print(probas)
    print(sum(predicted), sum(probas))
