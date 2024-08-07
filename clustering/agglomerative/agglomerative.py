from itertools import product
from operator import attrgetter
from typing import Set, Tuple

import numpy as np
import pandas as pd


class Point:
    def __init__(self, idx: int, coordinates: np.array):
        self.idx = idx
        self.coordinates = coordinates

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return self.idx == other.idx and self.coordinates == other.coordinates


class Cluster:
    def __init__(self, points: Tuple[Point, ...], created_at=None):
        self.points = points
        self.created_at = created_at or 0
        self._centroid = None

    @property
    def centroid(self) -> np.array:
        if self._centroid is None:
            if len(self.points) == 1:
                self._centroid = self.points[0].coordinates
            else:
                self._centroid = np.array([
                    np.mean(feature)
                    for feature in zip(*map(attrgetter('coordinates'), self.points))
                ])
        return self._centroid

    def union(self, other: 'Cluster'):
        return Cluster(
            points=self.points + other.points,
            created_at=min(self.created_at, other.created_at)
        )

    def __hash__(self):
        return hash(self.points)


class Metric:
    def __init__(self, name: str):
        self.name = name
        self._function = getattr(self, name)

    def get(self, *args):
        return self._function(*args)

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


class MyAgglomerative:
    def __init__(self, n_clusters: int = 3, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = Metric(metric)

    def fit_predict(self, X: pd.DataFrame):
        n = len(X)
        clusters: Set[Cluster] = {
            Cluster(points=(Point(idx=i, coordinates=X.iloc[i].values),))
            for i in range(n)
        }

        while len(clusters) > self.n_clusters:
            min_distance, cluster1, cluster2 = float('inf'), None, None

            for x, y in product(clusters, repeat=2):
                if x == y:
                    continue

                distance = self.metric.get(x.centroid, y.centroid)
                if distance < min_distance:
                    min_distance = distance
                    cluster1, cluster2 = x, y

            if cluster1 is None or cluster2 is None:
                break

            clusters.remove(cluster1)
            clusters.remove(cluster2)
            clusters.add(cluster1.union(cluster2))

        inverse_idx = [0] * n
        for i, cluster in enumerate(clusters, 1):
            for point in cluster.points:
                inverse_idx[point.idx] = i

        return inverse_idx

    def __str__(self):
        return f'{self.__class__.__name__} class: n_clusters={self.n_clusters}'


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]

    my_aggl = MyAgglomerative(n_clusters=10)
    print(my_aggl.fit_predict(X))
