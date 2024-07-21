from typing import Optional, Union

import numpy as np
import pandas as pd


class Metric:
    def __init__(self, name):
        self.name = name
        self._function = getattr(self, name)

    def get(self, y_fact, y_predicted):
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


class MyLineReg:
    def __init__(
        self,
        n_iter: int,
        learning_rate: float,
        weights: Optional[pd.Series] = None,
        metric: Optional[str] = None,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric and Metric(metric)
        self.last_metric_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        for i in range(self.n_iter):
            Y = np.dot(X, self.weights)
            mse = np.mean((Y - y) ** 2)
            if verbose is not False and i % verbose == 0:
                metric_info = ''
                if self.metric:
                    print(f' | {self.metric.name}: {self.metric.get(y, Y)}')

                print(f'{i} | loss: {mse}' + metric_info)

            grad_mse = 2 / n * np.dot((Y - y), X)
            self.weights -= self.learning_rate * grad_mse

        if self.metric:
            self.last_metric_value = self.metric.get(y, np.dot(X, self.weights))

    def predict(self, X: pd.DataFrame):
        X.insert(0, 'ones', 1)
        return np.dot(X, self.weights)

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.last_metric_value

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    my_line_reg = MyLineReg(400, 0.1, metric='mae')
    my_line_reg.fit(X, y, 50)

    X_test, y_test = make_regression(n_samples=1000, n_features=14, n_informative=5, noise=5)
    X_test = pd.DataFrame(X_test)

    # print(my_line_reg.predict(X_test))
    print(my_line_reg.get_best_score())
