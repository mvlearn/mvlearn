import numpy as np
from sklearn.base import BaseTransformer
from sklearn.utils.validation import check_is_fitted

from ..utils.utils import check_Xs


class ConcatTransformer(BaseTransformer):
    def __init__(self):
        pass

    def fit(self, Xs, y=None):
        self.n_features_ = [X.shape[1] for X in Xs]
        return self

    def transform(self, Xs, y=None):
        Xs = check_Xs(Xs)
        return np.hstack(Xs)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        return np.split(X, np.cumsum(self.n_features_)[:-1], axis=1)


class MeanTransformer(BaseTransformer):
    def __init__(self):
        pass

    def fit(self, Xs, y=None):
        check_Xs(Xs)
        n_features_ = [X.shape[1] for X in Xs]
        if len(set(n_features_)) > 1:
            raise ValueError('The number of features in each dataset should be'
                             'the same.')
        self.n_feature_ = n_features[1]
        return self

    def transform(self, Xs, y=None):
        Xs = check_Xs(Xs)
        return np.mean(Xs, axis=0)
