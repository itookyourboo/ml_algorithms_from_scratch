import random
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from linear_models.linear_regression.metric import Metric
from linear_models.linear_regression.regularization import Regularization


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
        sgd_sample: Union[int, float, None] = None,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric and Metric(metric)
        self.last_metric_value = None
        self.reg = reg and Regularization(reg, l1_coef, l2_coef)
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        random.seed(self.random_state)

        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        for i in range(1, self.n_iter + 1):
            Y = np.dot(X, self.weights)
            mse = np.mean((Y - y) ** 2)
            if self.reg:
                mse += self.reg.calculate(self.weights)

            self.log_if_verbose(verbose=verbose, iteration=i, loss=mse, metric_value=self.metric.calculate(y, Y))

            sample_X, sample_Y, sample_y = X, Y, y
            if isinstance(self.sgd_sample, (int, float)):
                n = self.sgd_sample
                if isinstance(self.sgd_sample, float):
                    n = int(X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), n)
                sample_X = X.iloc[sample_rows_idx]
                sample_Y = Y[sample_rows_idx]
                sample_y = y.iloc[sample_rows_idx]

            grad_mse = 2 / n * np.dot((sample_Y - sample_y), sample_X)
            if self.reg:
                grad_mse += self.reg.calculate_gradient(self.weights)
            self.weights -= self.get_learning_rate(i) * grad_mse

        if self.metric:
            self.last_metric_value = self.metric.calculate(y, np.dot(X, self.weights))

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

    def log_if_verbose(
        self,
        verbose: Union[int, bool],
        iteration: int,
        loss: float,
        metric_value: float,
    ):
        if verbose is not False and iteration % verbose == 0:
            metric_info = ''
            if self.metric:
                print(f' | {self.metric.name}: {metric_value}')

            print(f'{iteration} | loss: {loss}' + metric_info)

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
        sgd_sample=0.1,
    )
    my_line_reg.fit(X, y, 50)

    X_test, y_test = make_regression(n_samples=1000, n_features=14, n_informative=5, noise=5)
    X_test = pd.DataFrame(X_test)

    # print(my_line_reg.predict(X_test))
    print(my_line_reg.get_best_score())
