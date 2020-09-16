"""Filtering module."""
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

from sklearn.base import clone, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import check_Xs


class RepeatTransform(TransformerMixin):
    r"""Apply a sklearn transformer to each view of a dataset

    Build a transformer from multiview dataset to multiview dataset by
    using individual scikit-learn transformer on each view.

    Parameters
    ----------
    base_transformer : a sklearn transformer instance, or a list
        Either a single sklearn transformer that will be applied to each
        view. One clone of the estimator will correspond to each view.
        Otherwise, it should be a list of estimators, of length the number of
        views in the multiview dataset.

    Attributes
    ----------
    n_views_ : int
        The number of views in the input dataset

    transformers_ : list of objects of length n_views_
        The list of transformer used to transform data. If
        self.base_transformer is a single transformer, it is a list containing
        clones of that transformer, otherwise it is a view of
        self.base_transformer.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.preprocessing import RepeatTransform
    >>> from sklearn.decomposition import PCA
    >>> Xs, _ = load_UCImultifeature()
    >>> repeat = RepeatTransform(PCA(n_components=2))
    >>> Xs_transformed = repeat.fit_transform(Xs)
    >>> print(len(Xs_transformed))
    6
    >>> print(Xs_transformed[0].shape)
    (2000, 2)
    """

    def __init__(self, base_transformer):
        self.base_transformer = base_transformer

    def _prefit(self, Xs, y=None):
        r"""Estimate the attributes of the class.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Xs = check_Xs(Xs)
        self.n_views_ = len(Xs)
        if type(self.base_transformer) is list:
            if len(self.base_transformer) != self.n_views_:
                raise ValueError(
                    "The length of the transformers should be the same as the"
                    "number of views"
                )
            self.transformers_ = self.base_transformer
        else:
            self.transformers_ = [
                clone(self.base_transformer) for _ in range(self.n_views_)
            ]
        return self

    def fit(self, Xs, y=None):
        r"""Fit each estimator to the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._prefit(Xs, y)
        for transformer, X in zip(self.transformers_, Xs):
            transformer.fit(X)
        return self

    def fit_transform(self, Xs, y=None):
        r"""Fit and transform each dataset

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        self._prefit(Xs, y)
        Xs_transformed = []
        for transformer, X in zip(self.transformers_, Xs):
            Xs_transformed.append(transformer.fit_transform(X))
        return Xs_transformed

    def transform(self, Xs, y=None):
        r"""Transform each dataset

        Applies the transform of each transformer on the
        individual views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        y : None
            Ignored variable.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        Xs_transformed = []
        for transformer, X in zip(self.transformers_, Xs):
            Xs_transformed.append(transformer.transform(X))
        return Xs_transformed

    def inverse_transform(self, Xs, y=None):
        r"""Compute the inverse transform of a dataset

        Applies the inverse_transform function of each
        transformer on the individual datasets

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        y : None
            Ignored variable.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        Xs_transformed = []
        for transformer, X in zip(self.transformers_, Xs):
            Xs_transformed.append(transformer.inverse_transform(X))
        return Xs_transformed
