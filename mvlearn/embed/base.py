# License: MIT

from abc import abstractmethod
from numbers import Number
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from ..utils import check_Xs, param_as_list, svd_wrapper


class BaseEmbed(BaseEstimator):
    """
    A base class for embedding multiview data.
    Parameters
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs, y=None):
        """
        A method to fit model to multiview data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)

        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self

    @abstractmethod
    def transform(self, Xs):
        """
        Transform data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        Xs_transformed : list of numpy.ndarray
            - length: n_views
            - Xs_transformed[i] shape: (n_samples, n_components_i)
        """

        pass

    def fit_transform(self, Xs, y=None):
        """
        Fit an embedder to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y : array, shape (n_samples,), optional
            Targets to be used if fitting the algorithm is supervised.

        Returns
        -------
        X_transformed : list of numpy.ndarray
            - X_transformed length: n_views
            - X_transformed[i] shape: (n_samples, n_components_i)
        """
        return self.fit(Xs=Xs, y=y).transform(Xs=Xs)


class BaseCCA(BaseEstimator, TransformerMixin):
    """
    A base class for multiview CCA methods.
    """

    def fit(self, Xs, y=None):
        r"""
        Learns decompositions of the views.

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
        _, _ = self._fit(Xs)
        return self

    def transform(self, Xs):
        """
        Transform the views, projecting them using fitted loadings.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The views to transform

        Returns
        -------
        Xs_scores : numpy.ndarray, shape (n_views, n_samples, n_components)
            If `multiview_output`, returns the normed sum of transformed views
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        if len(Xs) != self.n_views_:
            msg = f"Supplied data must have {self.n_views_} views"
            raise ValueError(msg)
        scores = np.asarray([self.transform_view(X, i)
                             for i, X in enumerate(Xs)])
        if self.multiview_output:
            return scores
        else:
            common_scores = sum(scores)
            return common_scores / self.common_score_norms_

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
        X_scores : numpy.ndarray, shape (n_samples, n_components)
            Transformed view
        """
        check_is_fitted(self)
        X = check_array(X)
        if self.means_[view] is not None:
            X = X - self.means_[view]
        return X @ self.loadings_[view]

    def fit_transform(self, Xs, y=None):
        """
        Fit CCA to the data and transforms the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The views to fit and transform

        y : None
            Ignored variable.

        Returns
        -------
        Xs_scores : numpy.ndarray, shape (n_views, n_samples, n_components)
            If `multiview_output`, returns the normed sum of transformed views
        """
        scores, common_scores_normed = self._fit(Xs)
        if self.multiview_output:
            return scores
        else:
            return common_scores_normed


def _check_regs(regs, n_views):
    """
    Checks the regularization paramters for each view.
    If the regulaization is not None, it must be a float between 0 and 1

    Parameters
    ----------
    regs : float | 'lw' | 'oas' | None, or list, optional (default None)
        MCCA regularization for each data view, which can be important
        for high dimensional data. A list will specify for each view
        separately. If float, must be between 0 and 1 (inclusive).

        - 0 or None : corresponds to SUMCORR-AVGVAR MCCA.

        - 1 : partial least squares SVD (generalizes to more than 2 views)

        - 'lw' : Default ``sklearn.covariance.ledoit_wolf`` regularization

        - 'oas' : Default ``sklearn.covariance.oas`` regularization

    n_views : int
        Number of views

    Returns
    -------
    regs : list of parameters
    """
    regs = param_as_list(regs, n_views)
    for reg in regs:
        if reg is not None and isinstance(reg, Number):
            reg = float(reg)
            assert (reg >= 0) and (reg <= 1), \
                f"regs should be between 0 and 1, not {reg}"
        elif reg is not None and isinstance(reg, str):
            assert reg in ["oas", "lw"], \
                f'{reg} must be in ["oas", "lw"]'
    return regs


def _initial_svds(
    Xs,
    signal_ranks=None,
    normalized_scores=False,
    sval_thresh=None,
):
    """
    Computes a low rank SVD of each view in a list of data views.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The data to fit to.

    signal_ranks : None, int, list
        The initial signal rank to compute i.e. rank of the SVD.
        If None, will compute the full SVD.
        Different values can be provided for each view by inputting a list.

    normalized_scores : bool
        Whether or not to return the normalized scores matrix U as the
        primary output (left singular vectors) or the unnormalized scores
        i.e. UD.

    sval_thresh : None, float, or list
        Whether or not to theshold singular values i.e. delete SVD
        components whose singular value is below this threshold. A list
        will specify for each view separately.

    Returns
    -------
    reduced : list of array-like
        The left singular vectors of each view. If `normalized_scores` is
        True, then they are multiplied by the singular values.

    svds : list of tuples
        The low rank SVDs for each data view, (U, D, V) for X = UDV^T
    """

    Xs, n_views, _, _ = check_Xs(Xs, return_dimensions=True)
    signal_ranks = param_as_list(signal_ranks, n_views)
    sval_thresh = param_as_list(sval_thresh, n_views)

    # possibly perform SVDs on some views
    svds = [None] * n_views
    reduced = [None] * n_views
    for b in range(n_views):
        U, D, V = svd_wrapper(Xs[b], rank=signal_ranks[b])

        # possibly threshold SVD components
        if sval_thresh[b] is not None:
            to_keep = D >= sval_thresh[b]
            if sum(to_keep) == 0:
                raise ValueError(
                    f"all singular values of view {b} where thresholded at" +
                    f"{sval_thresh[b]}. Either this view is zero or you" +
                    "should try a smaller threshold value"
                )
            U = U[:, to_keep]
            D = D[to_keep]
            V = V[:, to_keep]
        svds[b] = U, D, V

        if normalized_scores:
            reduced[b] = U
        else:
            reduced[b] = U * D

    return reduced, svds


def _deterministic_decomp(common_scores, scores=None, loadings=None):
    """
    Enforces determinsitic decomposition output. Makes largest absolute value
    entry of common scores positive.
    """
    max_abs_cols = np.argmax(np.abs(common_scores), axis=0)
    signs = np.sign(common_scores[max_abs_cols, range(common_scores.shape[1])])
    common_scores = common_scores * signs
    for b in range(len(scores)):
        if scores is not None:
            scores[b] = scores[b] * signs
        if loadings is not None:
            loadings[b] = loadings[b] * signs

    return common_scores, scores, loadings
