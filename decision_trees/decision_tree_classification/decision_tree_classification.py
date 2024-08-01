from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


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


def split_data(X, feature, threshold, node_indices=None):
    if node_indices is None:
        node_indices = X.index

    left_indices, right_indices = [], []
    for i in node_indices:
        if X[feature][i] <= threshold:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return np.array(left_indices), np.array(right_indices)


def entropy(y):
    ans = 0
    n = len(y)
    for _, count in zip(*np.unique(y, return_counts=True)):
        p = count / n
        ans += -p * np.log2(p)
    return ans


def gini(y):
    sm = 0
    for _, count in zip(*np.unique(y, return_counts=True)):
        sm += count ** 2 / len(y) ** 2
    return 1 - sm


def gain(parent, left, right, ambiguity_func):
    parent_ambiguity, left_ambiguity, right_ambiguity = map(ambiguity_func, (parent, left, right))
    left_weight, right_weight = len(left) / len(parent), len(right) / len(parent)
    weighted_ambiguity = left_weight * left_ambiguity + right_weight * right_ambiguity
    return parent_ambiguity - weighted_ambiguity


def get_best_split(
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: pd.Series,
    gain_func,
    node_indices=None,
):
    if node_indices is None:
        node_indices = X.index

    best_split = {'feature': None, 'gain': -1, 'threshold': None}
    for feature in X:
        for threshold in thresholds[feature]:
            left_indices, right_indices = split_data(X, feature, threshold, node_indices)
            cur_gain = gain(y, y[left_indices], y[right_indices], gain_func)
            if cur_gain > best_split['gain']:
                best_split['feature'] = feature
                best_split['gain'] = cur_gain
                best_split['threshold'] = threshold

    return best_split['feature'], best_split['threshold'], best_split['gain']


class MyTreeClf:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
        criterion: str = 'entropy',
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.tree = None
        self.bins = bins

        assert criterion in ('entropy', 'gini')
        self.gain_func = [gini, entropy][criterion == 'entropy']

        self.fi = {}

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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fi = {feature: 0 for feature in X}
        _, self.tree = self.build_tree(
            X=X,
            y=y,
            node_indices=X.index,
            levels_left=self.max_depth,
            leaves_left=self.max_leafs,
        )

    def predict_proba(self, X: pd.DataFrame):
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

    def predict(self, X: pd.DataFrame):
        return self.predict_proba(X) > .5

    def build_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        node_indices: np.ndarray,
        levels_left: int,
        leaves_left: int,
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

        thresholds = self.prepare_thresholds(X)

        feature, threshold, cur_gain = get_best_split(X, y, thresholds, self.gain_func, node_indices)
        left_indices, right_indices = split_data(X, feature, threshold, node_indices)

        self.fi[feature] += len(node_indices) / len(X) * cur_gain

        l_leaves, l_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=left_indices,
            levels_left=levels_left - 1,
            leaves_left=leaves_left - 1,
        )
        r_leaves, r_tree = self.build_tree(
            X=X,
            y=y,
            node_indices=right_indices,
            levels_left=levels_left - 1,
            leaves_left=leaves_left - l_leaves,
        )

        tree = TreeNode(
            feature=feature,
            threshold=threshold,
            samples=len(node_indices),
            criterion=self.gain_func(y[node_indices]),
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
            f'max_leafs={self.max_leafs}'
        )


if __name__ == '__main__':
    # from sklearn.datasets import make_regression
    #
    # X_train, y_train = make_regression(n_samples=50, n_features=10, n_informative=2, random_state=42)
    # X_train = pd.DataFrame(X_train)
    # # X_train = pd.DataFrame(np.rint(100 * X_train))
    # y_train = pd.Series(np.rint(y_train % 1 < .5))
    # X_train.columns = [f'col_{col}' for col in X_train.columns]
    #
    # clf = MyTreeClf()
    # clf.fit(X_train, y_train)
    # clf.print_tree()
    # print(clf.leafs_cnt)

    from sklearn.model_selection import train_test_split
    df = pd.read_csv(
        '/Users/wignorbo/PycharmProjects/ml_algorithms_from_scratch/data_banknote_authentication.txt',
        header=None,
    )
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:, :4], df['target']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

    def get_sm(tree: TreeNode):
        if tree.is_leaf:
            return tree.value
        return get_sm(tree.left) + get_sm(tree.right)

    for i, (input_data, (leaves, sm)) in enumerate((
        ((1, 1, 2, 8, 'gini'), (2, 0.981148)),
        ((3, 2, 5, None, 'gini'), (5, 2.799994)),
        ((5, 200, 10, 4, 'entropy'), (10, 5.020575)),
        ((4, 100, 17, 16, 'gini'), (11, 5.200813)),
        ((10, 40, 21, 10, 'gini'), (21, 10.198869)),
    )):
        print(f'TEST #{i + 1}', input_data)
        clf = MyTreeClf(*input_data)
        clf.fit(X, y)
        clf.print_tree()

        print(f'LEAVES COUNT: {clf.leafs_cnt}. SHOULD BE: {leaves}')
        tree_sm = get_sm(clf.tree)
        print(f'LEAVES SUM: {tree_sm}. SHOULD BE: {sm}')
        print()
