"""Canonical Correlation Analysis"""

# Authors: Ronan Perry, Theodore Lee
# License: MIT

import numpy as np
from scipy.stats import f, chi2
from sklearn.utils.validation import check_is_fitted
from .mcca import MCCA, _i_mcca, _mcca_gevp
from ..utils import check_Xs, param_as_list


class CCA(MCCA):
    """Canonical Correlation Analysis (CCA)

    CCA inherits from MultiCCA (MCCA) but is restricted to 2 views which
    allows for certain statistics to be computed about the results.

    Parameters
    ----------
    n_components : int | 'min' | 'max' | None (default 1)
        Number of final components to compute. If `int`, will compute that
        many. If None, will compute as many as possible. 'min' and 'max' will
        respectively use the minimum/maximum number of features among views.

    regs : float | 'lw' | 'oas' | None, or list, optional (default None)
        MCCA regularization for each data view, which can be important
        for high dimensional data. A list will specify for each view
        separately. If float, must be between 0 and 1 (inclusive).

        - 0 or None : corresponds to SUMCORR-AVGVAR MCCA.

        - 1 : partial least squares SVD (generalizes to more than 2 views)

        - 'lw' : Default ``sklearn.covariance.ledoit_wolf`` regularization

        - 'oas' : Default ``sklearn.covariance.oas`` regularization

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
    MCCA, KMCCA

    References
    ----------
    .. [#1cca] Kettenring, J. R., "Canonical Analysis of Several Sets of
                Variables." Biometrika, 58 (1971), pp. 433-451
    .. [#2cca] Tenenhaus, A., et al. "Regularized generalized canonical
                correlation analysis." Psychometrika, 76(2):257.

    Examples
    --------
    >>> from mvlearn.embed import CCA
    >>> X1 = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> X2 = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA()
    >>> cca.fit([X1, X2])
    CCA()
    >>> Xs_scores = cca.transform([X1, X2])
    """

    def _fit(self, Xs):
        """Helper function for the `.fit` function"""
        Xs, self.n_views_, _, self.n_features_ = check_Xs(
            Xs, return_dimensions=True
        )
        if self.n_views_ != 2:
            raise ValueError(
                f"CCA accepts exactly 2 views but {self.n_views_}"
                "were provided. Consider using MCCA for more than 2 views")

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

    def stats(self, scores, stat=None):
        r"""
        Compute relevant statistics from the fitted CCA.

        Parameters
        ----------
        scores: array-like, shape (2, n_samples, n_components)
            The CCA scores.

        stat : str, optional (default None)
            The statistic to return. If None, returns a dictionary of all
            statistics. Otherwise, specifies one of the following statistics

            - 'r' : numpy.ndarray of shape (n_components,)
                Canonical correlations of each component.

            - 'Wilks' : numpy.ndarray of shape (n_components,)
                Wilks' Lambda likelihood ratio statistic.

            - 'df1' : numpy.ndarray of shape (n_components,)
                Degrees of freedom for the chi-squared statistic, and
                the numerator degrees of freedom for the F statistic.

            - 'df2' : numpy.ndarray of shape (n_components,)
                Denominator degrees of freedom for the F statistic.

            - 'F' : numpy.ndarray of shape (n_components,)
                Rao's approximate F statistic for H_0(k).

            - 'pF' : numpy.ndarray of shape (n_components,)
                Right-tail pvalue for stats['F'].

            - 'chisq' : numpy.ndarray of shape (n_components,)
                Bartlett's approximate chi-squared statistic for H_0(k)
                with Lawley's modification.

            - 'pChisq' : numpy.ndarray of shape (n_components,)
                Right-tail pvalue for stats['chisq'].

        Returns
        -------
        stats : dict or numpy.ndarray
            Dict containing the statistics with keys specified above or
            one of the statistics if specified by the `stat` parameter.
        """
        check_is_fitted(self)
        scores = check_Xs(scores, enforce_views=2)
        S1, S2 = scores
        assert S1.shape[1] == S2.shape[1], \
            "Scores from each view must have the same number of components."
        n_components = S1.shape[1]

        stats = {}

        # pearson correlation coefficient
        r = self.canon_corrs(scores)
        stats['r'] = r
        r = r.squeeze()

        # Wilks' Lambda test statistic
        d = min([n_components, min(self.n_features_)])
        k = np.arange(d)
        rank1_k = self.n_features_[0] - k
        rank2_k = self.n_features_[1] - k
        if r.size > 1:
            nondegen = np.argwhere(r < 1 - 2 * np.finfo(float).eps).squeeze()
        elif r < 1 - 2 * np.finfo(float).eps:
            nondegen = np.array(0, dtype=int)
        else:
            nondegen = np.array([], dtype=int)

        log_lambda = np.NINF * np.ones(n_components,)

        if nondegen.size > 0:
            if r.size > 1:
                log_lambda[nondegen] = np.cumsum(
                                        (np.log(1 - r[nondegen]**2))[::-1])
                log_lambda[nondegen] = log_lambda[nondegen][::-1]
            else:
                log_lambda[nondegen] = np.cumsum(
                                        (np.log(1 - r**2)))

        stats['Wilks'] = np.exp(log_lambda)

        # Rao's approximation to F distribution.
        # default value for cases where the exponent formula fails
        s = np.ones(d,)
        # cases where (d1k,d2k) not one of (1,2), (2,1), or (2,2)
        okCases = np.argwhere(rank1_k*rank2_k > 2).squeeze()
        snumer = rank1_k*rank1_k*rank2_k*rank2_k - 4
        sdenom = rank1_k*rank1_k + rank2_k*rank2_k - 5
        s[okCases] = np.sqrt(np.divide(snumer[okCases], sdenom[okCases]))

        # Degrees of freedom for null hypothesis H_0k
        stats['df1'] = rank1_k * rank2_k
        stats['df2'] = (
            S1.shape[0] - .5 * (self.n_features_[0] + self.n_features_[1] + 3)
            ) * s - (.5 * rank1_k * rank2_k) + 1

        # Rao's F statistic
        pow_lambda = stats['Wilks']**(1 / s)
        ratio = np.inf * np.ones(d,)
        ratio[nondegen] = ((1 - pow_lambda[nondegen]) / pow_lambda[nondegen])
        stats['F'] = ratio * stats['df2'] / stats['df1']

        # Right-tailed pvalue for Rao's F
        stats['pF'] = 1 - f.cdf(stats['F'], stats['df1'], stats['df2'])

        # Lawley's modification to Bartlett's chi-squared statistic
        if r.size == 1:
            r = np.array([r])
        stats['chisq'] = -log_lambda * (
            S1.shape[0] - k -
            0.5 * (self.n_features_[0] + self.n_features_[1] + 3) +
            np.cumsum(np.hstack((np.zeros(1,), 1 / r[: d-1]))**2))

        # Right-tailed pvalue for the Lawley modification to Barlett
        stats['pChisq'] = 1 - chi2.cdf(stats['chisq'], stats['df1'])

        if stat is None:
            return stats
        else:
            try:
                return stats[stat]
            except KeyError:
                raise KeyError(f"Provided statistic {stat} must be one of"
                               " the statistics listed in the Parameters.")
