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
# Authors: Pierre Ablin, Hugo Richard

import numpy as np
from picard import picard
import scipy.linalg as linalg
from sklearn.decomposition import fastica


from ..utils.utils import check_Xs
from .base import BaseEstimator
from .grouppca import GroupPCA


class GroupICA(BaseEstimator):
    r"""
    Group Independent component analysis.
    As an optional preprocessing, each dataset in `Xs` is reduced with
    usual PCA. Then, datasets are concatenated in the features direction,
    and a PCA is performed on this matrix, yielding a single dataset.
    Then an ICA is performed yielding the output dataset S. The unmixing matrix
    corresponding to data X are obtained by solving
    argmin_{W} ||S - WX||^2.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset

    n_individual_components : int or list of int or 'auto', optional
        The number of individual components to extract as a preprocessing.
        If None, no preprocessing is applied. If an `int`, each dataset
        is reduced to this dimension. If a list, the dataset `i` is
        reduced to the dimension `n_individual_components[i]`.
        If `'auto'`, set to the minimum between n_components and the
        smallest number of features in each dataset.

    multiple_outputs : bool, optional
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    prewhiten : bool, optional (default False)
        Whether the data should be whitened after the original preprocessing.

    solver : str {'fastica', 'picard'}
        Chooses which ICA solver to use. `picard` is generally faster and
        more reliable, but it requires to be installed.

    ica_kwargs : dict
        Optional keyword arguments for the ICA solver. If solver='fastica',
        see the documentation of sklearn.decomposition.fastica.
        If solver='picard', see the documentation of picard.picard.

    random_state : int, RandomState instance, default=None
            random state

    Attributes
    ----------
    means_ : list of arrays of shape (n_components,)
        The mean of each dataset

    grouppca_ : mvlearn.decomposition.GroupPCA instance
        A GroupPCA class for preprocessing and dimension reduction

    unmixing_ : array, shape (n_components, n_components)
        The square unmixing matrix, linking the output of the group-pca
        and the independent components.

    components_ : array, shape (n_components, n_components)
        The square mixing matrix

    individual_components_ : list of array
        Individual unmixing matrices estimated by least squares.
        `individual_components_[i]` is an array of shape
        (n_components, n_features) where n_features is the number of
        features in the dataset `i`.

    individual_mixing_ : list of array
        Individual mixing matrices estimated by least squares.
        `individual_components_[i]` is an array of shape
        (n_features, n_components) where n_features is the number of
        features in the dataset `i`.

    n_components_ : int
        The estimated number of components.

    n_features_ : list of int
        Number of features in each training dataset.

    n_samples_ : int
        Number of samples in the training data.

    n_subjects_ : int
        Number of subjects in the training data

    References
    ----------
    .. [#1groupica] Calhoun, Vince, et al. "A method for making group
                    inferences from functional MRI data using independent
                    component analysis."
                    Human brain mapping 14.3 (2001): 140-151.
    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import GroupICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = GroupICA(n_components=3)
    >>> Xs_transformed = ica.fit_transform(Xs)
    """

    def __init__(
        self,
        n_components=None,
        n_individual_components="auto",
        multiple_outputs=False,
        prewhiten=False,
        solver="fastica",
        ica_kwargs={},
        random_state=None,
    ):
        if solver == "picard":
            try:
                from picard import picard  # noqa: F401
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Picard does not seem to be "
                    "installed. Try $pip install "
                    "python-picard"
                )
        elif solver != "fastica":
            raise ValueError(
                "Invalid solver, must be either `fastica` or `picard`"
            )
        self.n_components = n_components
        self.n_individual_components = n_individual_components
        self.multiple_outputs = multiple_outputs
        self.prewhiten = prewhiten
        self.solver = solver
        self.ica_kwargs = ica_kwargs
        self.random_state = random_state

    def _fit(self, Xs, y=None):
        """
        Fit  to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            The transformed data
        """
        Xs = check_Xs(Xs, copy=True)
        self.means_ = [np.mean(X, axis=0) for X in Xs]
        gpca = GroupPCA(
            n_components=self.n_components,
            n_individual_components=self.n_individual_components,
            copy=True,
            prewhiten=self.prewhiten,
            whiten=True,
            random_state=self.random_state,
        )
        X_pca = gpca.fit_transform(Xs)
        self.grouppca_ = gpca
        if self.solver == "fastica":
            K, W, sources = fastica(
                X_pca, **self.ica_kwargs, random_state=self.random_state
            )
        else:
            K, W, sources = picard(
                X_pca.T, **self.ica_kwargs, random_state=self.random_state
            )
            sources = sources.T
        if K is not None:
            self.components_ = np.dot(W, K)
        else:
            self.components_ = W
        self.mixing_ = linalg.pinv(self.components_)
        # Compute individual unmixing matrices by least-squares
        self.individual_mixing_ = []
        self.individual_components_ = []
        sources_pinv = linalg.pinv(sources)
        for X, mean in zip(Xs, self.means_):
            lstq_solution = np.dot(sources_pinv, X - mean)
            self.individual_components_.append(linalg.pinv(lstq_solution).T)
            self.individual_mixing_.append(lstq_solution.T)
        self.n_components_ = gpca.n_components_
        self.n_features_ = gpca.n_features_
        self.n_samples_ = gpca.n_samples_
        self.n_subjects_ = gpca.n_subjects_
        return sources

    def fit_transform(self, Xs, y=None):
        """
        Fit  to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            The transformed data
        """
        sources = self._fit(Xs, y)
        if self.multiple_outputs:
            return self.transform(Xs)
        else:
            return sources

    def fit(self, Xs, y=None):
        r"""Fit the model with Xs.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(Xs, y)
        return self

    def transform(self, Xs, y=None):
        r"""
        A method to fit model to multiview data.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            The transformed data
        """
        Xs = check_Xs(Xs, copy=True)
        if self.multiple_outputs:
            return [
                np.dot(X - mean, W.T)
                for W, X, mean in (
                    zip(self.individual_components_, Xs, self.means_)
                )
            ]
        else:
            X = self.grouppca_.transform(Xs)
            return np.dot(X, self.components_.T)

    def inverse_transform(self, X_transformed):
        r"""
        A method to recover multiview data from transformed data
        """
        if self.multiple_outputs:
            return [
                np.dot(X, A.T) + mean
                for X, A, mean in (
                    zip(X_transformed, self.individual_mixing_, self.means_)
                )
            ]

        else:
            return self.grouppca_.inverse_transform(
                np.dot(X_transformed, self.mixing_.T)
            )
