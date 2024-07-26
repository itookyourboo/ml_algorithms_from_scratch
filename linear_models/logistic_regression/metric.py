from collections import defaultdict
from typing import Tuple

import numpy as np


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
