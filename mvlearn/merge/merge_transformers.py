"""Merging utilities."""
# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Authors: Pierre Ablin

import numpy as np

from abc import abstractmethod

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils.utils import check_Xs


class BaseMerger(TransformerMixin):
    """A base class for merging multiview datasets into single view datasets.

    The .merge function should return a single dataset. The .transform function
    should not be used, it is only included for compatibility with scikit learn
    pipelines.

    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    """
    def __init__(self):
        pass

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

        return self

    @abstractmethod
    def merge(self, Xs):
        r"""Merge multiview data

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

        pass

    def fit_merge(self, Xs, y=None):
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
        return self.fit(Xs, y).merge(Xs)

    @abstractmethod
    def inverse_merge(self, X):
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

        pass

    def transform(self, Xs, y=None):
        r"""Merge multiview data

        Necessary to be included into sklearn pipelines

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
        return self.merge(Xs, y)

    def fit_transform(self, Xs, y=None):
        r"""Fit  to the data and merge

        Necessary to be included into sklearn pipelines

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
        return self.fit_merge(Xs, y)

    @abstractmethod
    def inverse_transform(self, X):
        r"""Take a single view dataset and split it into multiple views.

        Necessary for inclusion in sklearn pipelines

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

        return self.inverse_merge(X)


class StackMerger(BaseMerger):
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
        Xs = check_Xs(Xs)
        self.n_features_ = [X.shape[1] for X in Xs]
        self.n_total_features_ = sum(self.n_features_)
        self.n_views_ = len(self.n_features_)
        return self

    def merge(self, Xs, y=None):
        r"""Merge the data by stacking its features.

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

    def inverse_merge(self, X):
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


class MeanMerger(BaseMerger):
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

    def merge(self, Xs, y=None):
        r"""Merge the views by averaging

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
