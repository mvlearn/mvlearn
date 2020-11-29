"""Merging utilities."""

# Authors: Pierre Ablin
#
# License: MIT

import numpy as np

from abc import abstractmethod

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils.utils import check_Xs


class BaseMerger(TransformerMixin):
    """A base class for merging multiview datasets into single view datasets.

    The .transform function should return a single dataset.

    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    """

    def __init__(self):
        pass  # pragma: no cover

    @abstractmethod
    def fit(self, Xs, y=None):
        r"""Fit model to multiview data.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self  # pragma: no cover

    @abstractmethod
    def transform(self, Xs, y=None):
        r"""Merge multiview data into a single dataset

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
             The singleview output
        """
        pass  # pragma: no cover

    def fit_transform(self, Xs, y=None):
        r"""Fit  to the data and merge

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_samples, n_features)
             The singleview output
        """
        return self.fit(Xs, y).transform(Xs)

    @abstractmethod
    def inverse_transform(self, X):
        r"""Take a single view dataset and split it into multiple views.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_total_features, n_samples)
            The input dataset

        Returns
        -------
        Xs : list of numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """

        pass  # pragma: no cover


class ConcatMerger(BaseMerger):
    r"""A transformer that stacks features of multiview datasets.

    Take a multiview dataset and transform it in a single view dataset
    by stacking features.

    Attributes
    ----------
    n_features_ : list of ints
        The number of features in each view of the input dataset

    n_total_features_ : int
        The number of features in the dataset, equal to the sum of n_features_

    n_views_ : int
        The number of views in the dataset

    See Also
    --------
    AverageMerger
    """

    def __init__(self):
        pass

    def fit(self, Xs, y=None):
        r"""Fit to the data.

        Stores the number of features in each view

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y
            Ignored

        Returns
        -------
        self : object
            Transformer instance.
        """
        Xs, n_views, n_samples, n_features = check_Xs(
            Xs, return_dimensions=True
        )
        self.n_features_ = n_features
        self.n_total_features_ = sum(self.n_features_)
        self.n_views_ = n_views
        return self

    def transform(self, Xs, y=None):
        r"""Merge the data by stacking its features.

        The multiple views are transformed into a single view dataset by
        stacking (i.e. concatenating) the features.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y
            Ignored

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_total_features, n_samples)
            The stacked data, containing all the stacked features.
        """
        Xs = check_Xs(Xs)
        return np.hstack(Xs)

    def inverse_transform(self, X):
        r"""Take a single view dataset and split it into multiple views.

        The input dimension must match the fitted dimension of the multiview
        dataset.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_total_features, n_samples)
            The input dataset

        Returns
        -------
        Xs : list of numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The multiview dataset obtained by splitting features of X
        """
        check_is_fitted(self)
        n_feature = X.shape[1]
        if n_feature != self.n_total_features_:
            raise ValueError(
                "The number of features in the input array ({}) does not match"
                " the total number of features in the multiview dataset"
                " ({})".format(n_feature, self.n_total_features_)
            )

        return np.split(X, np.cumsum(self.n_features_)[:-1], axis=1)


class AverageMerger(BaseMerger):
    r"""A transformer that computes the mean of multiview datasets

    Take a multiview dataset and transform it in a single view dataset
    by averaging across views


    Attributes
    ----------
    n_feature_ : list of ints
        The number of feature in each view of the input dataset
        Must be the same for each dataset.

    n_views_ : int
        The number of views in the dataset

    See Also
    --------
    ConcatMerger
    """

    def __init__(self):
        pass

    def fit(self, Xs, y=None):
        r"""Fit to the data.

        Stores the number of features in each view, and checks that
        each view has the same number of features.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y
            Ignored

        Returns
        -------
        self : object
            Transformer instance.
        """
        Xs = check_Xs(Xs)
        n_features_ = [X.shape[1] for X in Xs]
        if len(set(n_features_)) > 1:
            raise ValueError(
                "The number of features in each dataset should be the same."
            )
        self.n_feature_ = n_features_[0]
        self.n_views_ = len(n_features_)
        return self

    def transform(self, Xs, y=None):
        r"""Merge the views by averaging

        Transform the multiview dataset into a single view by averaging
        the views

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y
            Ignored

        Returns
        -------
        X_transformed : numpy.ndarray of shape (n_total_features, n_samples)
            The average of the views.
        """
        Xs = check_Xs(Xs)
        return np.mean(Xs, axis=0)
