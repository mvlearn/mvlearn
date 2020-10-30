"""Multiview Canonical Correlation Analysis"""

# Authors: Iain Carmichael, Ronan Perry
# License: MIT

from numbers import Number
import numpy as np
from scipy.linalg import block_diag
from itertools import combinations
from warnings import warn
from sklearn.covariance import ledoit_wolf, oas
from sklearn.utils.validation import check_is_fitted

from ..utils import check_Xs, param_as_list

from ..utils import eigh_wrapper, svd_wrapper
from .base import BaseCCA, _check_regs, _initial_svds, _deterministic_decomp
from ..compose import SimpleSplitter


class MCCA(BaseCCA):
    r"""Multiview CCA

    Multiview canonical correlation analysis. Includes options for
    regularized MCCA and informative MCCA (where a low rank PCA is first
    computed).

    Parameters
    ----------
    n_components : int | 'min' | 'max' | None (default 1)
        Number of final components to compute. If `int`, will compute that
        many. If None, will compute as many as possible. 'min' and 'max' will
        respectively use the minimum/maximum number of features among views.

    regs : float | 'lw' | 'oas' | None, or list, optional (default None)
        MCCA regularization for each data view, which can be important
        for high dimensional data. A list will specify for each view
        separately.

        - 0 | None: corresponds to SUMCORR-AVGVAR MCCA.

        - 1: partial least squares SVD in the case of 2 views and a natural
             generalization of this method for more than two views.

        - 'lw': Default ``sklearn.covariance.ledoit_wolf`` regularization

        - 'oas': Default ``sklearn.covariance.oas`` regularization

    signal_ranks : int, None or list, optional (default None)
        The initial signal rank to compute. If None, will compute the full SVD.
        A list will specify for each view separately.

    center : bool, or list (default True)
        Whether or not to initially mean center the data. A list will specify
        for each view separately.

    i_mcca_method : 'auto' | 'svd' | 'gevp' (default 'auto')
        Whether or not to use the SVD based method (only works with no
        regularization) or the gevp based method for informative MCCA.

    multiview_output : bool, optional (default True)
        If True, the ``.transform`` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    Attributes
    ----------
    means_ : list of numpy.ndarray
        The means of each view, each of shape (n_features,)

    loadings_ : list of numpy.ndarray
        The loadings for each view used to project new data,
        each of shape (n_features_b, n_components).

    common_score_norms_ : numpy.ndarray, shape (n_components,)
        Column norms of the sum of the fitted view scores.
        Used for projecting new data

    evals_ : numpy.ndarray, shape (n_components,)
        The generalized eigenvalue problem eigenvalues.

    n_views_ : int
        The number of views

    n_features_ : list
        The number of features in each fitted view

    n_components_ : int
        The number of components in each transformed view

    See also
    --------
    KMCCA

    References
    ----------
    .. [#1mcca] Kettenring, J. R., "Canonical Analysis of Several Sets of
                Variables." Biometrika, 58 (1971), pp. 433-451
    .. [#2mcca] Tenenhaus, A., et al. "Regularized generalized canonical
                correlation analysis." Psychometrika, 76(2):257.

    Examples
    --------
    >>> from mvlearn.embed import MCCA
    >>> X1 = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> X2 = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> mcca = MCCA()
    >>> mcca.fit([X1, X2])
    MCCA()
    >>> Xs_transformed = mcca.transform([X1, X2])
    """

    def __init__(
        self,
        n_components=1,
        regs=None,
        signal_ranks=None,
        center=True,
        i_mcca_method="auto",
        multiview_output=True,
    ):

        self.n_components = n_components
        self.center = center
        self.regs = regs
        self.signal_ranks = signal_ranks
        self.i_mcca_method = i_mcca_method
        self.multiview_output = multiview_output

    def _fit(self, Xs):
        """Helper function for the `.fit` function"""
        Xs, self.n_views_, _, self.n_features_ = check_Xs(
            Xs, return_dimensions=True
        )
        centers = param_as_list(self.center, self.n_views_)
        self.means_ = [np.mean(X, axis=0) if c else None
                       for X, c in zip(Xs, centers)]
        Xs = [X - m if m is not None else X for X, m in zip(Xs, self.means_)]

        if self.signal_ranks is not None:
            self.loadings_, scores, common_scores_normed, \
                self.common_score_norms_, self.evals_ = _i_mcca(
                    Xs,
                    signal_ranks=self.signal_ranks,
                    n_components=self.n_components,
                    regs=self.regs,
                    method=self.i_mcca_method,
                )
        else:
            self.loadings_, scores, common_scores_normed, \
                self.common_score_norms_, self.evals_ = _mcca_gevp(
                    Xs,
                    n_components=self.n_components,
                    regs=self.regs
                )
        return scores, common_scores_normed

    def inverse_transform(self, scores):
        """
        Transforms scores back to the original space.

        Parameters
        ----------
        scores: array-like, shape (n_samples, n_components)
            The CCA scores.

        Returns
        -------
        Xs_hat : list of array-likes or numpy.ndarray
             - Xs_hat length: n_views
             - Xs_hat[i] shape: (n_samples, n_features_i)
            The reconstructed original data
        """
        check_is_fitted(self)
        scores = check_Xs(scores)
        if len(scores) != self.n_views_:
            msg = f"Supplied data must have {self.n_views_} views"
            raise ValueError(msg)

        return [self.inverse_transform_view(score, i)
                for i, score in enumerate(scores)]

    def inverse_transform_view(self, scores, view):
        """
        Transforms scores back to the original space.

        Parameters
        ----------
        scores: array-like, shape (n_samples, n_components)
            The scores

        view : int
            The numeric index of the single view X with respect to the fitted
            views.

        Returns
        -------
        Xs_hat : numpy.ndarray, shape (n_samples, n_features)
            The reconstructed view
        """
        check_is_fitted(self)
        reconst = scores @ self.loadings_[view].T
        if self.means_[view] is not None:
            reconst += self.means_[view]
        return reconst

    def score(self, Xs, y=None):
        """
        Computes the squared reconstruction errors for all views.

            .. math::
                ||X_{hat} - X||_2^2

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
             The data to reconstruct and score against

        y : None
            Ignored variable.

        Returns
        -------
        scores : numpy.ndarray, shape (n_views,)
            Reconstruction scores. If ``self.multiview_output`` is True,
            then the mean score is returned.
        """
        Xs_hat = self.transform(Xs)
        Xs_hat = self.inverse_transform(Xs_hat)
        scores = [np.linalg.norm(X - X_hat) ** 2
                  for X, X_hat in zip(Xs, Xs_hat)]
        if self.multiview_output:
            return scores
        else:
            return np.mean(scores)

    def score_view(self, X, view):
        """
        Computes the squared reconstruction error for one view.

            .. math::
                ||X_{hat} - X||_2^2

        Parameters
        ----------
        Xs : numpy.ndarray, shape (n_samples, n_features)

        view : int
            The numeric index of the single view Xs with respect to the fitted
            views.

        Returns
        -------
        score : float
            Reconstruction score
        """
        check_is_fitted(self)
        X_hat = self.transform_view(X, view=view)
        X_hat = self.inverse_transform_view(X_hat, view=view)
        return np.linalg.norm(X - X_hat) ** 2

    @property
    def n_components_(self):
        if hasattr(self, "loadings_"):
            return self.loadings_[0].shape[1]
        else:
            raise AttributeError("Model has not been fitted properly yet")


def _mcca_gevp(Xs, n_components=None, regs=None):
    """
    Computes multi-view canonical correlation analysis via the generalized
    eigenvector formulation of SUMCORR-AVGVAR.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The data to fit to.

    n_components : None | int | 'min' | 'max'
        Number of final components to compute.

    regs : None | float | 'lw' | 'oas' or list of them, shape (n_views)
        MCCA regularization for each data view.

    Returns
    -------
    loadings : list of numpy.ndarray
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

    Xs, _, _, n_features = check_Xs(
        Xs, multiview=True, return_dimensions=True
    )
    n_components = _get_n_components(n_components, n_features)

    # solve generalized eigenvector problem
    LHS, RHS = _construct_mcca_gevp(Xs=Xs, regs=regs)
    try:
        gevals, ge_loadings = eigh_wrapper(A=LHS, B=RHS, rank=n_components)
    except np.linalg.LinAlgError as e:
        if 'is not positive definite' in str(e):
            raise ValueError(
                "Eigenvalue problem has a singular matrix. Add " +
                "regularization (set `regs` to nonzero value) or reduce " +
                "the rank (set `signal_rank` low enough).")
        else:
            raise e

    # Split rows
    loadings = [load.T for load in SimpleSplitter(n_features).fit_transform(
        ge_loadings.T)]
    scores = [X @ load for X, load in zip(Xs, loadings)]

    # common scores are the average of the view scores and are unit norm
    # this is also the flag mean of the subspaces spanned by the columns
    # of the views e.g. see (Draper et al., 2014)
    common_scores = sum(scores)
    common_norms = np.linalg.norm(common_scores, axis=0)
    common_norm_scores = common_scores / common_norms

    # enforce deterministic output due to possible sign flipsß
    common_scores_normed, scores, loadings = \
        _deterministic_decomp(common_norm_scores, scores,
                              loadings)

    return loadings, np.asarray(scores), common_scores_normed, \
        common_norms, gevals


def _i_mcca(Xs, signal_ranks=None, n_components=None, regs=None,
            method="auto"):
    """
    Computes informative multiview canonical correlation analysis
    e.g. PCA-CCA.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The data to fit to.

    signal_ranks: None, int, list
        The initial signal rank to compute i.e. rank of the SVD.
        If None, will compute the full SVD.
        Different values can be provided for each view by inputting a list.

    n_components : None | int | 'min' | 'max'
        Number of final components to compute.

    regs : None | float | 'lw' | 'oas' or list of them, shape (n_views)
        MCCA regularization for each data view.

    method: str, default='auto'
        Whether or not to use the SVD based method (only works with no
        regularization) or the gevp based method. Must be one of ['auto',
        'svd', 'gevp'].

    Returns
    -------
    loadings: list of numpy.ndarray
        The loadings for each view used to project new data,
        each of shape (n_features_b, n_components).

    scores: numpy.ndarray, shape (n_views, n_samples, n_components)
        Projections of each data view.

    common_scores_normed: numpy.ndarray, shape (n_samples, n_components)
        Normalized sum of the view scores.

    common_norms: numpy.ndarray, shape (n_components,)
        Column norms of the sum of the view scores.
        Useful for projecting new data

    gevals: numpy.ndarray, shape (n_components,)
        The generalized or flag subspace eigenvalues
    """
    Xs, n_views, n_samples, n_features = check_Xs(
        Xs, multiview=True, return_dimensions=True
    )
    regs = _check_regs(regs, n_views)

    if method == "auto":
        if regs is not None:
            method = "gevp"
        else:
            method = "svd"
    if method == "svd":
        assert all(r is None for r in regs), \
            "Regularization cannot be None for SVD method."

    # Compute initial SVD
    use_norm_scores = all(r is None for r in regs)

    bases, init_svds = _initial_svds(
        Xs=Xs, signal_ranks=signal_ranks, normalized_scores=use_norm_scores)

    # set n_components
    n_components = _get_n_components(
        n_components, [b.shape[1] for b in bases])

    # Compute MCCA on reduced data
    if method == "svd":
        # left singluar vectors for each view
        loadings, scores, common_scores_normed, gevals = \
            _flag_mean(bases, n_components=n_components)

        # map the view loadings back into the original feature space
        for b in range(n_views):
            D_b = init_svds[b][1]
            V_b = init_svds[b][2]
            W_b = V_b / D_b
            loadings[b] = W_b @ loadings[b]

        common_norms = np.sqrt(gevals)

    elif method == "gevp":
        # compute MCCA gevp problem on reduced data
        loadings, scores, common_scores_normed, common_norms, gevals = \
            _mcca_gevp(Xs=bases, n_components=n_components, regs=regs)

        # map the view loadings back into the original feature space
        for b in range(n_views):
            V_b = init_svds[b][2]
            if use_norm_scores:
                D_b = init_svds[b][1]
                W_b = V_b / D_b
            else:
                W_b = V_b
            loadings[b] = W_b @ loadings[b]

    return loadings, scores, common_scores_normed, common_norms, gevals


def _construct_mcca_gevp(Xs, regs=None, as_lists=False):
    r"""
    Constructs the matrices for the MCCA generalized eigenvector problem
    :math:`LHS v = \lambda RHS v`.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        The list of data matrices

    regs : None | float | 'lw' | 'oas' or list of them, shape (n_views)
        As described in ``mvlearn.mcca.mcca.MCCA``

    as_lists : bool
        If True, returns LHS and RHS as lists of composing blocks instead
        of their composition into full matrices.

    Returns
    -------
    LHS, RHS : numpy.ndarray, (sum_b n_features_b, sum_b n_features_b)
        Left and right hand side matrices for the GEVP
    """
    Xs, n_views, n_samples, n_features = check_Xs(
        Xs, multiview=True, return_dimensions=True
    )
    regs = _check_regs(regs, n_views)

    LHS = [[None for b in range(n_views)] for b in range(n_views)]
    RHS = [None for b in range(n_views)]

    # cross covariance matrices
    for (a, b) in combinations(range(n_views), 2):
        LHS[a][b] = Xs[a].T @ Xs[b]
        LHS[b][a] = LHS[a][b].T

    # view covariance matrices, possibly regularized
    for b in range(n_views):
        if regs[b] is None:
            RHS[b] = Xs[b].T @ Xs[b]
        elif isinstance(regs[b], Number):
            RHS[b] = (1 - regs[b]) * Xs[b].T @ Xs[b] + \
                regs[b] * np.eye(n_features[b])
        elif isinstance(regs[b], str):
            if regs[b] == "lw":
                RHS[b] = ledoit_wolf(Xs[b])[0]
            elif regs[b] == "oas":
                RHS[b] = oas(Xs[b])[0]
            # put back on scale of X^TX as oppose to
            # proper cov est returned by these functions
            RHS[b] *= n_samples

        LHS[b][b] = RHS[b]

    if not as_lists:
        LHS = np.block(LHS)
        RHS = block_diag(*RHS)

    return LHS, RHS


def _get_n_components(n_components, n_features):
    """Gets n_components from param and features"""
    if n_components is None:
        n_components = sum(n_features)
    elif n_components == "min":
        n_components = min(n_features)
    elif n_components == "max":
        n_components = max(n_features)
    elif n_components > sum(n_features):
        warn("Requested too many components. Setting to number of features")
        n_components = min(n_components, sum(n_features))
    return n_components


def _flag_mean(bases, n_components=None):
    """
    Computes the subspace flag mean.

    Parameters
    ----------
    bases: list
        List of orthonormal basis matrices for each subspace.

    n_components: None, int
        Number of components to compute.

    Returns
    -------
    loadings: list of numpy.ndarray
        The loadings for each view used to project new data,
        each of shape (n_features_b, n_components).

    scores: numpy.ndarray, shape (n_views, n_samples, n_components)
        Projections of each data view.

    flag_mean: nunpy.ndarray, (ambient_dim, n_components)
        Flag mean orthonormal basis matrix.

    sqsvals: numpy.ndarray, (n_components, )
        The squared singular values

    Notes
    -----
    Given a colletion of
    orthonormal matrices, X_1, ..., X_B we compute the the low rank SVD of
    X := [X_1, ..., X_B]. The left singular vectors are the flag mean. We
    refer to the right singular vectors as the "loadings". We further
    refer to the projection of each view onto its corresponding entries of
    the view loadings as the "scores".

    References
    ----------
    .. [#3mcca] Draper B., et al. "A flag representation for finite
                collections of subspaces of mixed dimensions."
                Linear Algebra Appl., 451 (2014), pp. 15-32
    """
    bases, n_views, ambient_dim, subspace_dims = check_Xs(
        bases, multiview=True, return_dimensions=True
    )

    # compte SVD of concatenated basis matrix
    flag_mean, svals, loadings = svd_wrapper(
        np.hstack(bases), rank=n_components)

    sqsvals = svals ** 2

    # get the view loadings and scores, split on rows
    loadings = [load.T for load in SimpleSplitter(
        subspace_dims).fit_transform(loadings.T)]
    scores = [b @ l for b, l in zip(bases, loadings)]

    flag_mean, scores, loadings = _deterministic_decomp(
        flag_mean, scores, loadings
    )

    return loadings, np.asarray(scores), flag_mean, sqsvals
