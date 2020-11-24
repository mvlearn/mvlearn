# BSD 3-Clause License
# Copyright (c) 2020, Hugo RICHARD and Pierre ABLIN
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Modified from source package https://github.com/hugorichard/multiviewica

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

try:
    from multiviewica import multiviewica
except ModuleNotFoundError as error:
    msg = (f"ModuleNotFoundError: {error}. multiviewica dependencies" +
           "required for this function. Please consult the mvlearn" +
           "installation instructions at https://github.com/mvlearn/mvlearn" +
           "to correctly install multiviewica dependencies.")
    raise ModuleNotFoundError(msg)

from .base import BaseDecomposer
from ..compose import ViewTransformer


class MultiviewICA(BaseDecomposer):
    r"""
    Multiview ICA for which views share a common source but separate mixing
    matrices.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset.

    noise : float, default=1.0
        Gaussian noise level

    max_iter : int, default=1000
        Maximum number of iterations to perform

    init : {'permica', 'groupica'} or np array of shape
        (n_groups, n_components, n_components), default='permica'
        If permica: initialize with perm ICA, if groupica, initialize with
        group ica. Else, use the provided array to initialize.

    preproc: 'pca' or a ViewTransformer-like instance,
        default='pca'
        Preprocessing method to use to reduce data.
        If "pca", performs PCA separately on each view to reduce dimension
        of each view.
        Otherwise the dimension reduction is performed using the transform
        method of the ViewTransformer-like object. This instance also needs
        an inverse transform method to recover original data from reduced data.

    multiview_output : bool, optional (default True)
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    random_state : int, RandomState instance or None, default=None
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.

    tol : float, default=1e-3
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.

    verbose : bool, default=False
        Print information

    Attributes
    ----------
    pcas_ : ViewTransformer instance
        The fitted `ViewTransformer` used to reduce the data.
        The `ViewTransformer` is given by
        `ViewTransformer(PCA(n_components=n_components))`
        where n_components is the number of chosen.
        Only used if n_components is not None.

    mixing_ : array, shape (n_views, n_components, n_components)
        The square mixing matrices, linking preprocessed data
        and the independent components.

    pca_components_: array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions
        of maximum variance in the data. Only used if n_components is not None.

    components_ : array, shape (n_views, n_components, n_components)
        The square unmixing matrices

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

    See also
    --------
    groupica

    Notes
    -----
    Given each view :math:`X_i` It optimizes:

        .. math::
            l(W) = \frac{1}{T} \sum_{t=1}^T [\sum_k log(cosh(Y_{avg,k,t}))
            + \sum_i l_i(X_{i,.,t})]

    where

        .. math::
            l _i(X_{i,.,t}) = - log(|W_i|) + 1/(2 \sigma) ||X_{i,.,t}W_i -
            Y_{avg,.,t}||^2,

    :math:`W_i` is the mixing matrix for view  :math:`i`,
    :math:`Y_{avg} = \frac{1}{n} \sum_{i=1}^n X_i W_i`, and :math:`\sigma`
    is the noise level.

    References
    ----------
    .. [#1mvica] Hugo Richard, Luigi Gresele, Aapo Hyvärinen, Bertrand Thirion,
        Alexandre Gramfort, Pierre Ablin. Modeling Shared Responses in
        Neuroimaging Studies through MultiView ICA. arXiv 2020.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import MultiviewICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = MultiviewICA(n_components=3, max_iter=10)
    >>> sources = ica.fit_transform(Xs)
    >>> print(sources.shape)
    (6, 2000, 3)
    """

    def __init__(
        self,
        n_components=None,
        noise=1.0,
        max_iter=1000,
        init="permica",
        multiview_output=True,
        random_state=None,
        tol=1e-3,
        verbose=False,
        n_jobs=30,
    ):
        self.verbose = verbose
        self.n_components = n_components
        self.noise = noise
        self.max_iter = max_iter
        self.init = init
        self.tol = tol
        self.random_state = random_state
        self.multiview_output = multiview_output

    def fit(self, Xs, y=None):
        """Fits the model.

        Parameters
        ----------
        Xs: list of np arrays of shape (n_voxels, n_samples)
            Input data: X[i] is the data of subject i

        y : ignored
        """
        if self.n_components is not None:
            self.pcas_ = ViewTransformer(
                PCA(n_components=self.n_components)
            )

        if self.n_components is not None:
            reduced_X = self.pcas_.fit_transform(Xs)
        else:
            reduced_X = Xs.copy()
        reduced_X = np.array(reduced_X)
        _, unmixings_, S = multiviewica(
            np.swapaxes(reduced_X, 1, 2),
            noise=self.noise,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
        )
        mixing_ = np.array([np.linalg.pinv(W) for W in unmixings_])
        self.components_ = unmixings_
        self.mixing_ = mixing_
        if self.n_components is not None:
            pca_components = []
            for i, estimator in enumerate(
                self.pcas_.estimators_
            ):
                K = estimator.components_
                pca_components.append(K)
            self.pca_components_ = np.array(pca_components)

        if self.n_components is None:
            self.individual_components_ = unmixings_
            self.individual_mixing_ = mixing_
        else:
            self.individual_mixing_ = []
            self.individual_components_ = []
            sources_pinv = linalg.pinv(S.T)
            for x in Xs:
                lstq_solution = np.dot(sources_pinv, x)
                self.individual_components_.append(
                    linalg.pinv(lstq_solution).T
                )
                self.individual_mixing_.append(lstq_solution.T)
        return self

    def transform(self, X):
        r"""
        Recover the sources from each view (apply unmixing matrix).

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Training data to recover a source and unmixing matrices from.

        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """

        if not hasattr(self, "components_"):
            raise ValueError("The model has not yet been fitted.")

        if self.n_components is not None:
            transformed_X = self.pcas_.transform(X)
        else:
            transformed_X = X.copy()
        if self.multiview_output:
            return np.array(
                [x.dot(w.T) for w, x in zip(self.components_, transformed_X)]
            )
        else:
            return np.mean(
                [x.dot(w.T) for w, x in zip(self.components_, transformed_X)],
                axis=0,
            )

    def inverse_transform(self, X_transformed):
        r"""
        Transforms the sources back to the mixed data for each view
        (apply mixing matrix).

        Parameters
        ----------
        X_transformed : list of array-likes or numpy.ndarray
            The dataset corresponding to transformed data
        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """
        check_is_fitted(self, "components_")
        if self.multiview_output:
            S_ = np.mean(X_transformed, axis=0)
        else:
            S_ = X_transformed
        inv_red = [S_.dot(w.T) for w in self.mixing_]

        if self.n_components is not None:
            return self.pcas_.inverse_transform(inv_red)
        else:
            return inv_red
