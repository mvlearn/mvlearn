"""Kernel Multiview Canonical Correlation Analysis"""
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
# Authors: Iain Carmichael, Ronan Perry

import numpy as np
from itertools import combinations
from warnings import warn
from sklearn.metrics import pairwise_kernels
from ..utils import check_Xs, param_as_list
from ..compose import SimpleSplitter
from ..utils import eigh_wrapper
from .base import BaseCCA, _check_regs, _initial_svds, _deterministic_decomp
from sklearn.utils.validation import check_is_fitted


class KMCCA(BaseCCA):
    r"""
    Kernel multi-view canonical correlation analysis.

    Parameters
    -----------
    n_components : int, None (default 1)
        Number of components to compute. If None, will use the number of
        features.

    kernel : str, callable, or list (default 'linear')
        The kernel function to use. This is the metric argument to
        ``sklearn.metrics.pairwise.pairwise_kernels``. A list will
        specify for each view separately.

    kernel_params : dict, or list (default {})
        Key word arguments to ``sklearn.metrics.pairwise.pairwise_kernels``.
        A list will specify for each view separately.

    regs : float, None, or list, optional (default None)
        None equates to 0. Floats are nonnegative. The value is used to
        regularize singular values in each view based on `diag_mode`
        A list will specify the method for each view separately.

    signal_ranks : int, None, or list, optional (default None)
        Largest SVD rank to compute for each view. If None, the full rank
        decomposition will be used. A list will specify for each view
        separately.

    sval_thresh : float, or list (default 1e-3)
        For each view we throw out singular values of (1/n)K, the gram matrix
        scaled by n_samples, below this threshold. A non-zero value deals with
        singular gram matrices.

    diag_mode : 'A' | 'B' | 'C' (default 'A')
        Method of regularizing singular values `s` with regularization
        parameter `r`

        - 'A' : :math:`(1 - r) * K^2 + r * K` [#1kmcca]_

        - 'B' : :math:`(1-r) (K + n/2 \kappa * I)^2` where
          :math:`\kappa = r / (1 - r)` [#2kmcca]_

        - 'C' : :math:`(1 - r) K^2 + r * I_n` [#3kmcca]_

    center : bool, or list (default True)
        Whether or not to initially mean center the data. A list will
        specify for each view separately.

    filter_params : bool (default False)
        See ``sklearn.metrics.pairwise.pairwise_kernels`` documentation.

    n_jobs : int, None, optional (default None)
        Number of jobs to run in parallel when computing kernel matrices.
        See ``sklearn.metrics.pairwise.pairwise_kernels`` documentation.

    multiview_output : bool, optional (default True)
        If True, the ``.transform`` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    Attributes
    ----------
    kernel_col_means_ : list of numpy.ndarray, shape (n_samples,)
        The column means of each gram matrix

    kernel_mat_means_ : list
        The total means of each gram matrix

    dual_vars_ : numpy.ndarray, shape (n_views, n_samples, n_components)
        The loadings for the gram matrix of each view

    common_score_norms_ : numpy.ndarray, shape (n_components,)
        Column norms of the sum of the view scores.
        Useful for projecting new data

    evals_ : numpy.ndarray, shape (n_components,)
        The generalized eigenvalue problem eigenvalues.

    Xs : list of numpy.ndarray, length (n_views,)
        - Xs[i] shape (n_samples, n_features_i)
        The original data matrices for use in gram matrix computation
        during calls to ``.transform``.

    n_views_ : int
        The number of views

    n_features_ : list
        The number of features in each fitted view

    n_components_ : int
        The number of components in each transformed view

    See also
    --------
    MCCA

    References
    ----------
    .. [#1kmcca] Hardoon D., et al. "Canonical Correlation Analysis: An
                 Overview with Application to Learning Methods", Neural
                 Computation, Volume 16 (12), pp 2639-2664, 2004.
    .. [#2kmcca] Bach, F. and Jordan, M. "Kernel Independent Component
                 Analysis." JMLR, 3(Jul):1-48, 2002.
    .. [#3kmcca] Kuss, M. and Graepel, T.. "The Geometry of Kernel Canonical
                 Correlation Analysis." MPI Technical Report, 108. (2003).

    Examples
    --------
    >>> from mvlearn.embed import KMCCA
    >>> X1 = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> X2 = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> kmcca = KMCCA()
    >>> kmcca.fit([X1, X2])
    KMCCA()
    >>> Xs_transformed = kmcca.transform([X1, X2])
    """

    def __init__(
        self,
        n_components=1,
        kernel="linear",
        kernel_params={},
        regs=None,
        signal_ranks=None,
        sval_thresh=1e-3,
        diag_mode="A",
        center=True,
        filter_params=False,
        n_jobs=None,
        multiview_output=True,
    ):

        self.n_components = n_components
        self.center = center
        self.regs = regs
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.signal_ranks = signal_ranks
        self.sval_thresh = sval_thresh
        self.diag_mode = diag_mode
        self.filter_params = filter_params
        self.n_jobs = n_jobs
        self.multiview_output = multiview_output

    def _fit(self, Xs):
        """Helper method for `.fit` function"""
        Xs, self.n_views_, _, self.n_features_ = check_Xs(
            Xs, multiview=True, return_dimensions=True)

        centers = param_as_list(self.center, self.n_views_)

        # set up (centered) kernels
        self.kernel_col_means_ = [None] * self.n_views_
        self.kernel_mat_means_ = [None] * self.n_views_
        Ks = []

        for b in range(self.n_views_):
            K = self._get_kernel(Xs[b], view=b)
            if centers[b]:
                K, col_mean, mat_mean = _center_kernel(K)
                self.kernel_col_means_[b] = col_mean
                self.kernel_mat_means_[b] = mat_mean
            Ks.append(K)

        self.dual_vars_, scores, common_scores_normed, \
            self.common_score_norms_, self.evals_ = _kmcca_gevp(
                Ks,
                signal_ranks=param_as_list(self.signal_ranks, self.n_views_),
                sval_thresh=self.sval_thresh,
                n_components=self.n_components,
                regs=self.regs,
                diag_mode=self.diag_mode,
            )

        del Ks
        self.Xs_ = Xs  # stored for `transform` step

        return scores, common_scores_normed

    def transform_view(self, X, view):
        """
        Transform a view, projecting it using fitted loadings.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            The view to transform

        view : int
            The numeric index of the single view X with respect to the fitted
            views.

        Returns
        -------
        X_transformed : numpy.ndarray, shape (n_samples, n_components)
            Transformed view
        """
        check_is_fitted(self)
        K = self._get_kernel(self.Xs_[view], X)
        if self.kernel_col_means_[view] is not None:
            row_means = np.sum(K, axis=1)[:, np.newaxis] / K.shape[0]
            K -= self.kernel_col_means_[view]
            K -= row_means
            K += self.kernel_mat_means_[view]
        return np.dot(K, self.dual_vars_[view])

    def _get_kernel(self, X, view, Y=None):
        """
        Returns a gram (kernel) matrix between a set of observations and
        itself or a second set of observations.

        Parameters
        ----------
        Xs : numpy.ndarray, shape (n, d)
            The data matrix

        view : int
            The view index, for the kernel parameter selection

        Y : numpy.ndarray, shape (m, d) (optional)
            Second data matrix

        Returns
        -------
        K : numpy.ndarray, shape (n, n) or (n, m) if Y provided
            The gram matrix with entries from the kernel function
        """
        if isinstance(self.kernel, list):
            kernel = self.kernel[view]
        else:
            kernel = self.kernel

        if isinstance(self.kernel_params, list):
            kernel_params = self.kernel_params[view]
        else:
            kernel_params = self.kernel_params

        return pairwise_kernels(
            X, Y, metric=kernel, filter_params=True, n_jobs=self.n_jobs,
            **kernel_params)

    @property
    def n_components_(self):
        if hasattr(self, "loadings_"):
            return self.dual_vars_[0].shape[1]
        else:
            raise AttributeError("Model has not been fitted properly yet")


def _kmcca_gevp(
    Ks,
    signal_ranks=None,
    n_components=None,
    regs=None,
    sval_thresh=1e-3,
    diag_mode="A",
):
    """
    Computes kernel MCCA using the eigenvector formulation where we compute
    matrix square roots via SVDs. By throwing out zero singular values of
    the kernel views we can avoid singularity issues.

    Parameters
    ----------
    Ks : list of numpy.ndarray, length (n_views,)
        - Ks[i] shape (n_samples, n_samples)
        List of kernel matrices

    n_components : int, None
        Number of components to compute. If None, will use the number of
        features.

    regs : float, None, or list
        None equates to 0. Floats are nonnegative. The value is used to
        regularize singular values in each view based on `diag_mode`
        A list will specify the method for each view separately.

    signal_ranks : int, None, or list
        Largest SVD rank to compute for each view. If None, the full rank
        decomposition will be used. A list will specify for each view
        separately.

    sval_thresh : float, (default 1e-3)
        For each view we throw out singular values of (1/n)K, the gram matrix
        scaled by n_samples, below this threshold. A non-zero value deals with
        singular gram matrices.

    diag_mode : 'A' | 'B' | 'C' (default 'A')
        Method of regularizing singular values `s` with regularization
        parameter `r`

        - 'A' : (1 - r) * K^2 + r * K [#1kmcca]_

        - 'B' : (1-r) (K + n/2 kappa * I)^2 where kappa = r / (1 - r)
          [#2kmcca]_

        - 'C' : (1 - r) K^2 + r * I [#3kmcca]_

    Returns
    -------
    dual_vars : numpy.ndarray, shape (n_views, n_samples, n_components)
        The loadings for each view used to project new data,
        each of shape (n_features_b, n_components).

    scores : numpy.ndarray, shape (n_views, n_samples, n_components)
        Projections of each data view.

    common_scores_normed : numpy.ndarray, shape (n_samples, n_components)
        Normalized sum of the view scores.

    common_norms : numpy.ndarray, shape (n_components,)
        Column norms of the sum of the view scores.
        Useful for projecting new data

    gevals : numpy.ndarray, shape (n_components,)
        The generalized eigenvalue problem eigenvalues.
    """
    if sval_thresh is not None:
        # put sval_thresh on the scale of svals K.
        sval_thresh *= Ks[0].shape[0]

    Us, svds = _initial_svds(Ks,
                             signal_ranks=signal_ranks,
                             normalized_scores=True,
                             sval_thresh=sval_thresh)
    svals = [svd[1] for svd in svds]

    Us, n_views, n_samples, n_features_reduced = check_Xs(
        Us, return_dimensions=True)
    regs = _check_regs(regs=regs, n_views=n_views)

    if n_components is None:
        n_components = sum(n_features_reduced)
    if n_components > sum(n_features_reduced):
        warn("Requested too many components!")
    n_components = min(n_components, sum(n_features_reduced))

    # get singular values of transformed view gram matrices
    reg_svals = _regularize_svals(
        svals=svals, regs=regs, diag_mode=diag_mode, n_samples=n_samples
    )

    # constructe matrix for eigen decomposition
    C = [[None for _ in range(n_views)] for _ in range(n_views)]
    for b in range(n_views):
        C[b][b] = np.eye(n_features_reduced[b])

    for (a, b) in combinations(range(n_views), 2):
        U_a = Us[a]
        s_a = svals[a]
        t_a = reg_svals[a]
        q_a = s_a * (1 / np.sqrt(t_a))

        U_b = Us[b]
        s_b = svals[b]
        t_b = reg_svals[b]
        q_b = s_b * (1 / np.sqrt(t_b))

        C[a][b] = (U_a * q_a).T @ (U_b * q_b)
        C[b][a] = C[a][b].T

    C = np.block(C)

    gevals, gevecs = eigh_wrapper(A=C, rank=n_components)
    gevecs = [gevec.T for gevec in SimpleSplitter(
        n_features_reduced).fit_transform(gevecs.T)]

    dual_vars = [(Us[b] / np.sqrt(reg_svals[b])) @ gevecs[b]
                 for b in range(n_views)]

    scores = [Ks[b] @ dual_vars[b] for b in range(n_views)]
    common_scores = sum(scores)
    common_norms = np.linalg.norm(common_scores, axis=0)
    common_scores_normed = common_scores / common_norms

    # enforce deterministic output due to possible sign flips
    common_scores_normed, scores, dual_vars = _deterministic_decomp(
        common_scores_normed, scores, dual_vars)

    return np.asarray(dual_vars), np.asarray(scores), common_scores_normed, \
        common_norms, gevals


def _regularize_svals(svals, regs=None, diag_mode="A",
                      n_samples=None):
    """
    Regularizes singular values for various mode options.

    Parameters
    ----------
    svals : list of numpy.ndarray
        Singular values of each kernel matrix.

    regs : None, float, or list
        Regularization parameter for each view.

    diag_mode : 'A' | 'B' | 'C'
        Method of regularizing singular values `s` with regularization
        parameter `r`

    n_samples : None, int
        Number of samples. Needed for mode B.

    Returns
    -------
    reg_svals : list of array-like
        Regularized singular values for each view and given diag_mode.
    """
    n_views = len(svals)

    reg_svals = [None for b in range(n_views)]
    for b in range(n_views):
        s = np.array(svals[b])

        if regs is None:
            r = None
        else:
            r = regs[b]

        if r is None or r == 0:
            t = s ** 2
        elif diag_mode == "A":
            t = (1 - r) * s ** 2 + r * s
        elif diag_mode == "B":
            assert n_samples is not None
            if np.isclose(r, 1):
                t = s
            else:
                kappa = r / (1 - r)
                t = (1 - r) * (s + 0.5 * n_samples * kappa) ** 2
        elif diag_mode == "C":
            t = (1 - r) * s ** 2 + r

        reg_svals[b] = t

    return reg_svals


def _center_kernel(K):
    """
    Centers a kernel matrix data.

    Parameters
    ----------
    K : np.ndarray, shape (n,n)
        A kernel matrix

    Returns
    -------
    K_c : numpy.ndarray, shape (n,n)
        The centered kernel matrix

    col_mean : numpy.ndarray, shape (n,)
        Mean of each kernel matrix column

    mat_mean : float
        Mean of entire kernel matrix
    """

    col_mean = np.mean(K, axis=0)
    mat_mean = np.mean(col_mean)
    K_c = K - col_mean - col_mean[:, np.newaxis] + mat_mean

    return K_c, col_mean, mat_mean
