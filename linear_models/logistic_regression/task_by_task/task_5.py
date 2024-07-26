from collections import defaultdict
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class Metric:
    def __init__(self, name):
        self.name = name
        self._function = getattr(self, name)

    def get(self, y_fact, y_predicted, y_predicted_proba=None):
        return self._function(y_fact, y_predicted, y_predicted_proba)

    @classmethod
    def _get_matrix(cls, y_fact, y_predicted) -> Tuple[int, int, int, int]:
        tp = tn = fp = fn = 0
        for actual, predicted in zip(y_fact, y_predicted):
            tp += actual == 1 == predicted
            tn += actual == 0 == predicted
            fp += actual == 0 != predicted
            fn += actual == 1 != predicted
        return tp, tn, fp, fn

    @classmethod
    def accuracy(cls, y_fact, y_predicted, _):
        tp, tn, fp, fn = cls._get_matrix(y_fact, y_predicted)
        return (tp + tn) / (tp + tn + fp + fn)

    @classmethod
    def precision(cls, y_fact, y_predicted, _):
        tp, tn, fp, fn = cls._get_matrix(y_fact, y_predicted)
        return tp / (tp + fp)

    @classmethod
    def recall(cls, y_fact, y_predicted, _):
        tp, tn, fp, fn = cls._get_matrix(y_fact, y_predicted)
        return tp / (tp + fn)

    @classmethod
    def f1(cls, y_fact, y_predicted, _):
        precision = cls.precision(y_fact, y_predicted, _)
        recall = cls.recall(y_fact, y_predicted, _)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def roc_auc(cls, y_fact, _, y_predicted_proba):
        P = N = 0
        cnt = defaultdict(lambda: [0, 0])
        for proba, kls in zip(y_predicted_proba, y_fact):
            kls = int(kls)
            P += kls == 1
            N += kls == 0
            cnt[np.round(proba, 10)][int(kls)] += 1

        sm = 0
        positives = 0
        for proba in sorted(cnt.keys(), reverse=True):
            if cnt[proba][0]:
                sm += cnt[proba][0] * (positives + cnt[proba][1] / 2)
            positives += cnt[proba][1]

        return sm / P / N



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


class MyLogReg:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float = 0.1,
        weights: Optional[np.ndarray] = None,
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

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(xi) for xi in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        z = np.exp(x)
        return z / (1 + z)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        X = X.copy()
        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        eps = 1e-15
        for epoch in range(1, self.n_iter + 1):
            Y = self._sigmoid(X @ self.weights)
            log_loss = -sum(
                y[i] * np.log(Y[i] + eps) + (1 - y[i]) * np.log(1 - Y[i] + eps)
                for i in range(n)
            ) / n
            if self.reg:
                log_loss += self.reg.get(self.weights)

            if verbose and epoch % verbose == 0:
                metric_info = ''
                if self.metric:
                    metric_info = f' | {self.metric.name}: {self.metric.get(y, Y > 0.5, Y)}'
                print(f'{epoch} | loss: {log_loss}' + metric_info)

            log_loss_grad = 1 / n * (Y - y) @ X
            if self.reg:
                log_loss_grad += self.reg.get_grad(self.weights)

            self.weights -= self.learning_rate * log_loss_grad

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

    my_log_reg = MyLogReg(n_iter=50, learning_rate=0.1, metric='roc_auc')
    my_log_reg.fit(X_train, y_train, 5)

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
