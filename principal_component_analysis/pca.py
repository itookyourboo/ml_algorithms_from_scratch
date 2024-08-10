import numpy as np
import pandas as pd


class MyPCA:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def fit_transform(self, X: pd.DataFrame):
        X_meaned = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_meaned, rowvar=False)

        values, vectors = np.linalg.eigh(cov_matrix)
        indices = np.argsort(values)[-self.n_components:][::-1]
        principal_components = vectors[:, indices]

        return X_meaned @ principal_components

    def __str__(self):
        return f'{self.__class__.__name__} class: n_components={self.n_components}'
