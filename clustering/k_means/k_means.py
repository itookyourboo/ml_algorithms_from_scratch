from collections import defaultdict

import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 10,
        n_init: int = 3,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = float('inf')

    @staticmethod
    def _distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    def init_centroids(self, X: pd.DataFrame):
        return [
            np.array([
                np.random.uniform(X[feature].min(), X[feature].max())
                for feature in X.columns
            ])
            for _ in range(self.n_clusters)
        ]

    def get_distribution(self, X, centroids):
        distribution_by_point = defaultdict(lambda: (-1, float('inf')))
        for cluster_num in range(self.n_clusters):
            if len(centroids[cluster_num]) == 0:
                continue

            for index, point in X.iterrows():
                distance = self._distance(point, centroids[cluster_num])
                if distance < distribution_by_point[index][1]:
                    distribution_by_point[index] = (cluster_num, distance)

        wcss = 0
        distribution_by_cluster = [[] for _ in range(self.n_clusters)]
        for point_index, (cluster, distance) in distribution_by_point.items():
            distribution_by_cluster[cluster].append(X.values[point_index])
            wcss += distance ** 2

        return list(map(np.array, distribution_by_cluster)), wcss

    def fit(self, X: pd.DataFrame):
        np.random.seed(seed=self.random_state)
        best_wcss, best_centroids = float('inf'), None

        for _ in range(self.n_init):
            centroids = self.init_centroids(X)
            wcss = float('inf')
            for _ in range(self.max_iter):
                distribution, wcss = self.get_distribution(X, centroids)
                new_centroids = []
                for cluster in range(self.n_clusters):
                    coords = []
                    for feature_values in zip(*distribution[cluster]):
                        coords.append(np.mean(feature_values))
                    new_centroids.append(coords or centroids[cluster])

                if all(all(np.isclose(v1, v2) for v1, v2 in zip(c1, c2)) for c1, c2 in zip(centroids, new_centroids)):
                    break
                centroids = new_centroids

            if wcss < best_wcss:
                best_wcss = wcss
                best_centroids = centroids

        self.cluster_centers_ = best_centroids
        self.inertia_ = best_wcss

    def predict(self, X: pd.DataFrame):
        ans = []
        for point in X.values:
            best_distance, best_centroid_num = float('inf'), -1
            for i, centroid in enumerate(self.cluster_centers_, 1):
                distance = self._distance(point, centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_centroid_num = i
            ans.append(best_centroid_num)
        return ans

    def __str__(self):
        args = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__} class: {args}'


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
    X = pd.DataFrame(X)
    X.columns = [f'col_{col}' for col in X.columns]

    k_means = MyKMeans(n_clusters=10, max_iter=10, n_init=3)
    k_means.fit(X)
    print((round(k_means.inertia_, 10), np.array(k_means.cluster_centers_).sum().round(10)))