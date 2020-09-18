"""Splitting utilities."""
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
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils.utils import check_Xs


class BaseSplitter(TransformerMixin):
    """A base class for splitting single view datasets into multiview datasets

    The .transform function should return a multiview dataset

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
    def fit(self, X, y=None):
        r"""Fit model to multiview data.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data

        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self  # pragma: no cover

    @abstractmethod
    def transform(self, X, y=None):
        r"""Split singleview dataset into multiple views

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data

        y : array, shape (n_samples,), optional

        Returns
        -------
        Xs_transformed : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        """
        pass  # pragma: no cover

    def fit_transform(self, X, y=None):
        r"""Fit to the data and split

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Input data

        y : array, shape (n_samples,), optional

        Returns
        -------
        Xs_transformed : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def inverse_transform(self, Xs):
        r"""Take a multiview dataset and merge it into a single view dataset

        Parameters
        ----------
        Xs : list of numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        X : numpy.ndarray, shape (n_total_features, n_samples)
            The input dataset

        """

        pass  # pragma: no cover


class ConcatSplitter(BaseSplitter):
    r"""A transformer that splits the features of a single dataset.

    Take a singleview dataset and transform it in a multiview dataset
    by splitting features to different views

    Parameters
    ----------
    n_features : list of ints
        The number of feature to keep in each split: Xs[i] will have shape
        (n_samples, n_features[i])

    Attributes
    ----------
    n_total_features_ : int
        The number of features in the dataset, equal to the sum of n_features_

    n_views_ : int
        The number of views in the output dataset

    See Also
    --------
    ConcatMerger
    """
    def __init__(self, n_features):
        self.n_features = n_features

    def fit(self, X, y=None):
        r"""Fit to the data.

        Checks that X has a compatible shape.

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y
            Ignored

        Returns
        -------
        self : object
            Transformer instance.
        """
        X = check_array(X)
        _, n_total_features = X.shape
        self.n_total_features_ = sum(self.n_features)
        if self.n_total_features_ != n_total_features:
            raise ValueError("The number of features of X should equal the sum"
                             " of n_features")
        self.n_views_ = len(self.n_features)
        return self

    def transform(self, X, y=None):
        r"""Split data

        The singleview dataset and transform it in a multiview dataset
        by splitting features to different views

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y
            Ignored

        Returns
        -------
        Xs_transformed : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.split(X, np.cumsum(self.n_features)[:-1], axis=1)

    def inverse_transform(self, Xs):
        r"""Take a multiview dataset and merge it in a single view

        The input dimension must match the fitted dimension of the multiview
        dataset.

        Parameters
        ----------
        Xs : list of numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The input multiview dataset

        Returns
        -------
        X : numpy.ndarray, shape (n_total_features, n_samples)
            The output singleview dataset
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        for X, n_feature in zip(Xs, self.n_features):
            if X.shape[1] != n_feature:
                raise ValueError("The number of features in Xs does not match"
                                 " n_features")

        return np.hstack(Xs)
