from _operator import itemgetter


class Weight(Getter):
    @classmethod
    def uniform(cls, points):
        return sum(map(itemgetter(1), points)) / len(points)

    @classmethod
    def rank(cls, points):
        nom = sum(value / rank for rank, (_, value) in enumerate(points, 1))
        denom = sum(1 / x for x in range(1, len(points) + 1))
        return nom / denom

    @classmethod
    def distance(cls, points):
        nom = sum(value / distance for distance, value in points)
        denom = sum(1 / distance for distance, _ in points)
        return nom / denom
