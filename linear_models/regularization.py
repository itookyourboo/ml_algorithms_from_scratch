import numpy as np


class Regularization:
    def __init__(self, reg: str, l1_coef: float, l2_coef: float):
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self._function = getattr(self, reg)
        self._function_grad = getattr(self, f'{reg}_grad')

    def calculate(self, weights):
        return self._function(self.l1_coef, self.l2_coef, weights)

    def calculate_gradient(self, weights):
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