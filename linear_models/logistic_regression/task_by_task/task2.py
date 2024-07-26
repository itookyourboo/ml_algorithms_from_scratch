from typing import Optional, Union

import numpy as np
import pandas as pd


class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1, weights: Optional[np.ndarray] = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(xi) for xi in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        z = np.exp(x)
        return z / (1 + z)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
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
            if verbose and epoch % verbose == 0:
                print(epoch, '|', 'loss:', log_loss)
            log_loss_grad = 1 / n * (Y - y) @ X
            self.weights -= self.learning_rate * log_loss_grad

    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    my_log_reg = MyLogReg(n_iter=50, learning_rate=0.1)
    my_log_reg.fit(X, y, 5)
    print(my_log_reg.get_coef())