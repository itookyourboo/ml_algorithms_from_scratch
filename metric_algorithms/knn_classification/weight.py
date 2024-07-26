from operator import itemgetter

from metric_algorithms.knn_classification.utils import Getter


class Weight(Getter):
    @classmethod
    def uniform(cls, points):
        return sum(map(itemgetter(1), points)) / len(points)

    @classmethod
    def rank(cls, points):
        nom = sum(1 / rank for rank, (_, kls) in enumerate(points, 1) if kls == 1)
        denom = sum(1 / x for x in range(1, len(points) + 1))
        return nom / denom

    @classmethod
    def distance(cls, points):
        nom = sum(1 / distance for distance, kls in points if kls == 1)
        denom = sum(1 / distance for distance, _ in points)
        return nom / denom
