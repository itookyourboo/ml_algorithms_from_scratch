from typing import Union, Optional

import numpy as np
import pandas as pd


class MyLineReg:
    def __init__(self, n_iter: int, learning_rate: float, weights: Optional[pd.Series] = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        X.insert(0, 'ones', 1)
        feature_count = X.shape[1]

        self.weights = self.weights or np.ones(feature_count)

        n = y.shape[0]
        for i in range(self.n_iter):
            Y = np.dot(X, self.weights)
            mse = np.mean((Y - y) ** 2)
            if verbose is not False and i % verbose == 0:
                print(i, '|', 'loss:', mse)
            grad_mse = 2 / n * np.dot((Y - y), X)
            self.weights -= self.learning_rate * grad_mse

    def get_coef(self):
        return self.weights[1:]

    def __str__(self):
        return f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    my_line_reg = MyLineReg(50, 0.1)
    my_line_reg.fit(X, y, 5)
    print(my_line_reg.get_coef())
