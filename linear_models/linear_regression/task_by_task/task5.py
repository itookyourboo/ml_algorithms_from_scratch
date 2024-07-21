from typing import Callable, Optional, Union

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


class Regularization:
    def __init__(self, reg: str, l1_coef: float, l2_coef: float):
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self._function = getattr(self, reg)
        self._function_grad = getattr(self, f'{reg}_grad')

    def get(self, weights):
        return self._function(self.l1_coef, self.l2_coef, weights)

    def get_grad(self, weights):
        return self._function_grad(self.l1_coef, self.l2_coef, weights)

    @classmethod
    def l1(cls, l1_coef, _, weights):
        return l1_coef * np.sum(weights)

    @classmethod
    def l1_grad(cls, l1_coef, _, weights):
        return l1_coef * np.sign(weights)

    @classmethod
    def l2(cls, _, l2_coef, weights):
        return l2_coef * np.sum(weights ** 2)

    @classmethod
    def l2_grad(cls, _, l2_coef, weights):
        return 2 * l2_coef * weights

    @classmethod
    def elasticnet(cls, l1_coef, l2_coef, weights):
        return cls.l1(l1_coef, ..., weights) + cls.l2(..., l2_coef, weights)

    @classmethod
    def elasticnet_grad(cls, l1_coef, l2_coef, weights):
        return cls.l1_grad(l1_coef, ..., weights) + cls.l2_grad(..., l2_coef, weights)


class MyLineReg:
    def __init__(
        self,
        n_iter: int,
        learning_rate: Union[float, Callable],
        weights: Optional[pd.Series] = None,
        metric: Optional[str] = None,
        reg: Optional[str] = None,
        l1_coef: float = 0,
        l2_coef: float = 0,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric and Metric(metric)
        self.last_metric_value = None
        self.reg = reg and Regularization(reg, l1_coef, l2_coef)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        for i in range(1, self.n_iter + 1):
            Y = np.dot(X, self.weights)
            mse = np.mean((Y - y) ** 2)
            if self.reg:
                mse += self.reg.get(self.weights)
            if verbose is not False and i % verbose == 0:
                metric_info = ''
                if self.metric:
                    print(f' | {self.metric.name}: {self.metric.get(y, Y)}')

                print(f'{i} | loss: {mse}' + metric_info)

            grad_mse = 2 / n * np.dot((Y - y), X)
            if self.reg:
                grad_mse += self.reg.get_grad(self.weights)
            self.weights -= self.get_learning_rate(i) * grad_mse

        if self.metric:
            self.last_metric_value = self.metric.get(y, np.dot(X, self.weights))

    def get_learning_rate(self, i: int):
        if callable(self.learning_rate):
            return self.learning_rate(i)
        return self.learning_rate

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

    my_line_reg = MyLineReg(
        n_iter=400,
        learning_rate=lambda i: 0.5 * (0.85 ** i),
        # learning_rate=0.03,
        metric='mae',
        reg='elasticnet',
        l1_coef=0.5,
        l2_coef=0.5,
    )
    my_line_reg.fit(X, y, 50)

    X_test, y_test = make_regression(n_samples=1000, n_features=14, n_informative=5, noise=5)
    X_test = pd.DataFrame(X_test)

    # print(my_line_reg.predict(X_test))
    print(my_line_reg.get_best_score())
