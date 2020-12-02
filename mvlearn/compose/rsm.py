"""Random Subspace Method"""

# Authors: Ronan Perry
#
# License: MIT

import numpy as np
from sklearn.utils import check_array
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import check_n_views, check_n_features


class RandomSubspaceMethod(TransformerMixin):
    """
    Random Subspace Method [#1RSM]_ for constructing multiple views.
    Each view is constructed by randomly selecting features from X.

    Parameters
    ----------
    n_views : int
        Number of views to construct

    subspace_dim : int, float
        Number of features from the data to subsample. If float, is between 0
        and 1 and denotes the proportion of features to subsample.

    random_state : int or RandomState instance, optional (default None)
        Controls the random sampling of features. Set for
        reproducible results.

    Attributes
    ----------
    n_features_  : list, length n_views
        The number of features in the fitted data

    subspace_dim_ : int
        The number of features subsampled in each view

    subspace_indices_ : list of numpy.ndarray, length n_views
        Feature indices to subsample for each view

    References
    ----------
    .. [#1RSM] Tin Kam Ho. "The random subspace method for constructing
            decision forests." IEEE trans-actions on pattern analysis and
            machine intelligence, 20(8):832–844, 1998.

    .. [#2RSM] Dacheng Tao, Xiaoou Tang, Xuelong Li and Xindong Wu,
               "Asymmetric bagging and random subspace for support vector
               machines-based relevance feedback in image retrieval," in
               IEEE Transactions on Pattern Analysis and Machine Intelligence,
               28(7):1088-1099, July 2006

    Examples
    --------
    >>> from mvlearn.compose import RandomSubspaceMethod
    >>> import numpy as np
    >>> X = np.random.rand(1000, 50)
    >>> rsm = RandomSubspaceMethod(n_views=3, subspace_dim=10)
    >>> Xs = rsm.fit_transform(Xs)
    >>> print(len(Xs))
    3
    >>> print(Xs[0].shape)
    (1000, 10)
    """
    def __init__(self, n_views, subspace_dim, random_state=None):
        check_n_views(n_views)
        self.n_views = n_views
        self.subspace_dim = subspace_dim
        self.random_state = random_state

    def fit(self, X, y=None):
        r"""
        Fit to the singleview data.

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y : Ignored

        Returns
        -------
        self : object
            The Transformer instance
        """
        X = check_array(X)
        _, n_features = X.shape
        self.n_features_ = n_features
        check_n_features(self.subspace_dim, n_features)

        # check if n_feaures is between 0 and 1
        self.subspace_dim_ = self.subspace_dim
        if self.subspace_dim_ < 1:
            self.subspace_dim_ = int(self.subspace_dim_ * n_features)

        np.random.seed(self.random_state)
        self.subspace_indices_ = [
            np.random.choice(
                n_features, size=self.subspace_dim_, replace=False)
            for _ in range(self.n_views)]

        return self

    def transform(self, X):
        r"""
        Transforms the singleview dataset and into a multiview dataset.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input dataset

        Returns
        -------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, subspace_dim)
        """
        check_is_fitted(self)
        X = check_array(X)
        _, n_features = X.shape

        if n_features != self.n_features_:
            raise ValueError("Number of features different " +
                             f"than fitted number {self.n_features_}")

        Xs = [X[:, idxs] for idxs in self.subspace_indices_]

        return Xs
