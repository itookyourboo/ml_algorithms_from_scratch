import random
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class Metric:
    def __init__(self, name):
        self.name = name
        self._function = getattr(self, name)

    def calculate(self, y_fact, y_predicted):
        return self._function(y_fact, y_predicted)

    @classmethod
    def mae(cls, y_fact, y_predicted):
        return np.mean(np.abs(y_fact - y_predicted))

    @classmethod
    def mse(cls, y_fact, y_predicted):
        return np.mean((y_fact - y_predicted) ** 2)

    @classmethod
    def rmse(cls, y_fact, y_predicted):
        return np.sqrt(cls.mse(y_fact, y_predicted))

    @classmethod
    def r2(cls, y_fact, y_predicted):
        y_mean = np.mean(y_fact)
        return 1 - np.sum((y_fact - y_predicted) ** 2) / np.sum((y - y_mean) ** 2)

    @classmethod
    def mape(cls, y_fact, y_predicted):
        return 100 * np.mean(np.abs((y_fact - y_predicted) / y_fact))


class TreeNode:
    def __init__(
        self,
        value: Union[float, int, None] = None,
        feature: Optional[str] = None,
        threshold: Optional[float] = None,
        samples: Optional[int] = None,
        criterion: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
    ):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.samples = samples
        self.criterion = criterion
        self.left = left
        self.right = right

    @property
    def label(self):
        if self.is_leaf:
            return self.value
        return f'{self.feature} > {self.threshold}'

    @property
    def is_leaf(self):
        return not self.left and not self.right

    def print(self, depth=0):
        prefix = '-' * depth
        if self.is_leaf:
            print(f'{prefix} {self.value} ({self.samples})')
        else:
            print(f'{prefix} {self.feature} > {self.threshold} | '
                  f'criterion: {self.criterion}, samples: {self.samples}')
            if self.left:
                self.left.print(depth + 1)
            if self.right:
                self.right.print(depth + 1)


def mse(y):
    ym = np.mean(y)
    return np.mean((y - ym) ** 2)


def split_data(X, feature, threshold, node_indices=None):
    if node_indices is None:
        node_indices = X.index

    left_indices = node_indices[X.loc[node_indices, feature] <= threshold]
    right_indices = node_indices[X.loc[node_indices, feature] > threshold]
    return np.array(left_indices), np.array(right_indices)


def get_best_split(
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: pd.Series,
    node_indices=None,
):
    if node_indices is None:
        node_indices = X.index

    best_split = {
        'feature': None,
        'gain': -1,
        'threshold': None,
        'left_indices': None,
        'right_indices': None,
    }

    parent = y[node_indices]
    parent_mse = mse(parent)

    for feature in X.columns:
        for threshold in thresholds[feature]:
            left_indices, right_indices = split_data(X, feature, threshold, node_indices)
            if not len(left_indices) or not len(right_indices):
                continue

            left = y[left_indices]
            right = y[right_indices]
            left_mse, right_mse = map(mse, (left, right))
            left_weight, right_weight = len(left) / len(parent), len(right) / len(parent)
            cur_gain = parent_mse - (left_weight * left_mse + right_weight * right_mse)

            if cur_gain > best_split['gain']:
                best_split['feature'] = feature
                best_split['gain'] = cur_gain
                best_split['threshold'] = threshold
                best_split['left_indices'] = left_indices
                best_split['right_indices'] = right_indices

    return best_split


class MyTreeReg:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(2, max_leafs)
        self.leafs_cnt = 0
        self.bins = bins
        self.fi = {}

        self._thresholds = None

    def prepare_thresholds(self, X: pd.DataFrame):
        if self.bins and self._thresholds is not None:
            return self._thresholds

        thresholds_df = pd.Series()

        for feature in X:
            thresholds = np.unique(X[feature])
            if not self.bins or len(thresholds) - 1 <= self.bins:
                thresholds_df[feature] = np.array([
                    (thresholds[i] + thresholds[i + 1]) / 2
                    for i in range(len(thresholds) - 1)
                ])
            else:
                _, thresholds = np.histogram(X[feature], bins=self.bins)
                thresholds_df[feature] = thresholds[1:-1]

        if self.bins:
            self._thresholds = thresholds_df
        return thresholds_df

    def fit(self, X: pd.DataFrame, y: pd.Series, y_size=None):
        self.fi = {feature: 0 for feature in X.columns}
        _, self.tree = self.build_tree(
            X=X,
            y=y,
            node_indices=X.index,
            levels_left=self.max_depth,
            leaves_left=self.max_leafs,
            y_size=y_size,
        )

    def predict(self, X: pd.DataFrame):
        y = np.zeros(X.shape[0])
        i = 0
        for _, x in X.iterrows():
            tree: TreeNode = self.tree
            while not tree.is_leaf:
                if x[tree.feature] <= tree.threshold:
                    tree = tree.left
                else:
                    tree = tree.right
            y[i] = tree.value
            i += 1
        return y

    def build_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        node_indices: np.ndarray,
        levels_left: int,
        leaves_left: int,
        y_size=None,
    ) -> Tuple[int, TreeNode]:
        split_possible = len(node_indices) >= 1 and y[node_indices].nunique() > 1
        growth_limits = levels_left <= 0 or len(node_indices) < self.min_samples_split or leaves_left <= 1
        if not split_possible or growth_limits:
            self.leafs_cnt += 1
            return 1, TreeNode(
                value=np.mean(y[node_indices]),
                criterion=0,
                samples=len(node_indices),
            )

        y_size = y_size or len(y)
        thresholds = self.prepare_thresholds(X.loc[node_indices])
        best_split = get_best_split(X, y, thresholds, node_indices)
        self.fi[best_split['feature']] += len(node_indices) / y_size * best_split['gain']

        l_leaves, l_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=best_split['left_indices'],
            levels_left=levels_left - 1,
            leaves_left=leaves_left - 1,
            y_size=y_size,
        )
        r_leaves, r_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=best_split['right_indices'],
            levels_left=levels_left - 1,
            leaves_left=leaves_left - l_leaves,
            y_size=y_size,
        )

        tree = TreeNode(
            feature=best_split['feature'],
            threshold=best_split['threshold'],
            samples=len(node_indices),
            criterion=best_split['gain'],
            left=l_tree,
            right=r_tree,
        )

        return l_leaves + r_leaves, tree

    def print_tree(self):
        self.tree.print()

    def __str__(self):
        return (
            f'{self.__class__.__name__} class: '
            f'max_depth={self.max_depth}, '
            f'min_samples_split={self.min_samples_split}, '
            f'max_leafs={self.max_leafs}, '
            f'bins={self.bins}'
        )


class MyForestReg:
    def __init__(
        self,
        n_estimators: int = 10,
        max_features: float = 0.5,
        max_samples: float = 0.5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: int = 16,
        random_state: int = 42,
        oob_score: Optional[str] = None
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}
        self.oob_score = oob_score and Metric(oob_score)
        self.oob_score_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fi = {feature: 0 for feature in X}
        random.seed(self.random_state)
        init_cols = list(X.columns)
        init_rows_cnt = len(X)
        cols_smpl_cnt = round(self.max_features * len(init_cols))
        rows_smpl_cnt = round(self.max_samples * init_rows_cnt)

        oob_table = {}

        for tree_num in range(self.n_estimators):
            cols_idx = random.sample(init_cols, cols_smpl_cnt)
            rows_idx = random.sample(range(init_rows_cnt), rows_smpl_cnt)
            tree_reg = MyTreeReg(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_leafs=self.max_leafs,
                bins=self.bins,
            )
            tree_reg.fit(X.loc[rows_idx, cols_idx], y[rows_idx], y_size=init_rows_cnt)

            if self.oob_score:
                rows_oob = list(set(range(init_rows_cnt)) - set(rows_idx))
                y_pred = tree_reg.predict(X.loc[rows_oob])
                for i in range(len(rows_oob)):
                    oob_table.setdefault(rows_oob[i], []).append(y_pred[i])

            self.leafs_cnt += tree_reg.leafs_cnt
            for feature in tree_reg.fi:
                self.fi[feature] += tree_reg.fi[feature]
            self.trees.append(tree_reg)

        if self.oob_score:
            oob_indexes = list(oob_table.keys())
            self.oob_score_ = self.oob_score.calculate(
                y_fact=y.loc[oob_indexes],
                y_predicted=np.array([np.mean(oob_table[idx]) for idx in oob_indexes]),
            )

    def predict(self, X: pd.DataFrame):
        return sum(tree.predict(X) for tree in self.trees) / self.n_estimators

    def print_tree(self):
        for i, tree in enumerate(self.trees, 1):
            print(tree)
            tree.print_tree()

    def __str__(self):
        args = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__} class: {args}'


if __name__ == '__main__':
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    reg = MyForestReg(**{"n_estimators": 5, "max_depth": 4, "max_features": 0.4, "max_samples": 0.3})
    reg.fit(X, y)
    reg.print_tree()
    print(reg.leafs_cnt)
