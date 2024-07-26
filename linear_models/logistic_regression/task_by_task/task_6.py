import random
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from linear_models.linear_regression.metric import Metric
from linear_models.regularization import Regularization


class MyLogReg:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: Union[float, Callable] = 0.1,
        weights: Optional[np.ndarray] = None,
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

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(xi) for xi in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        z = np.exp(x)
        return z / (1 + z)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        random.seed(self.random_state)

        X = X.copy()
        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        eps = 1e-15
        for epoch in range(1, self.n_iter + 1):
            Y = self._sigmoid(X @ self.weights)
            log_loss = -sum(
                y.iloc[i] * np.log(Y[i] + eps) + (1 - y.iloc[i]) * np.log(1 - Y[i] + eps)
                for i in range(n)
            ) / n
            if self.reg:
                log_loss += self.reg.get(self.weights)

            if verbose and epoch % verbose == 0:
                metric_info = ''
                if self.metric:
                    metric_info = f' | {self.metric.name}: {self.metric.get(y, Y > 0.5, Y)}'
                print(f'{epoch} | loss: {log_loss}' + metric_info)

            sample_X, sample_Y, sample_y = X, Y, y
            if isinstance(self.sgd_sample, (int, float)):
                n = self.sgd_sample
                if isinstance(self.sgd_sample, float):
                    n = int(X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), n)
                sample_X = X.iloc[sample_rows_idx]
                sample_Y = Y[sample_rows_idx]
                sample_y = y.iloc[sample_rows_idx]

            log_loss_grad = 1 / n * (sample_Y - sample_y) @ sample_X
            if self.reg:
                log_loss_grad += self.reg.get_grad(self.weights)

            self.weights -= self.get_learning_rate(epoch) * log_loss_grad

        if self.metric:
            Y = self._sigmoid(X @ self.weights)
            self.last_metric_value = self.metric.get(y, Y > 0.5, Y)

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'ones', 1)
        Y = self._sigmoid(X @ self.weights)
        return Y > 0.5

    def predict_proba(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, 'ones', 1)
        return self._sigmoid(X @ self.weights)

    def get_learning_rate(self, i: int):
        if callable(self.learning_rate):
            return self.learning_rate(i)
        return self.learning_rate

    def get_best_score(self):
        return self.last_metric_value

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X_train, y_train = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train % 1 < 0.5)
    X_train.columns = [f'col_{col}' for col in X_train.columns]

    my_log_reg = MyLogReg(
        n_iter=50,
        learning_rate=0.1,
        metric='roc_auc',
        sgd_sample=0.1,
    )
    my_log_reg.fit(X_train, y_train, verbose=False)

    X_test, y_test = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test % 1 < 0.5)
    X_test.columns = [f'col_{col}' for col in X_test.columns]

    # print(my_log_reg.predict(X_test))
    # print(my_log_reg.predict_proba(X_test))
    print(my_log_reg.get_best_score())
    # print(Metric.roc_auc(
    #     y_fact=np.random.random(11) > 0.5,
    #     y_predicted=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    #     y_predicted_proba=[.91, .86, .78, .6, .6, .55, .51, .46, .45, .45, .42]
    # ))
