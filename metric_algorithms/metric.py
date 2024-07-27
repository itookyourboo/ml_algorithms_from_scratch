import numpy as np

from metric_algorithms.utils import Getter


class Metric(Getter):
    @classmethod
    def euclidean(cls, x1, x2):
        return np.linalg.norm(x1 - x2)

    @classmethod
    def chebyshev(cls, x1, x2):
        return np.max(np.abs(x1 - x2))

    @classmethod
    def manhattan(cls, x1, x2):
        return np.sum(np.abs(x1 - x2))

    @classmethod
    def cosine(cls, x1, x2):
        return 1 - np.sum(x1 * x2) / np.linalg.norm(x1) / np.linalg.norm(x2)
