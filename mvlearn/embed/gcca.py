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

from .base import BaseEmbed
from ..utils.utils import check_Xs

import numpy as np
from scipy import linalg, stats
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from .utils import select_dimension
import warnings


def center(X):
    r"""
    Subtracts the row means and divides by the row standard deviations.
    Then subtracts column means.

    Parameters
    ----------
    X : array-like, shape (n_observations, n_features)
        The data to preprocess

    Returns
    -------
    centered_X : preprocessed data matrix
    """

    # Mean along rows using sample mean and sample std
    centered_X = stats.zscore(X, axis=1, ddof=1)
    # Mean along columns
    mu = np.mean(centered_X, axis=0)
    centered_X -= mu
    return centered_X


class GCCA(BaseEmbed):
    r"""
    An implementation of Generalized Canonical Correlation Analysis [#1GCCA]_
    suitable for cases where the number of features exceeds the number of
    samples by first applying single view dimensionality reduction. Computes
    individual projections into a common subspace such that the correlations
    between pairwise projections are minimized (ie. maximize pairwise
    correlation). An important note: this is applicable to any number of
    views, not just two.

    Parameters
    ----------
    n_components : int (positive), optional, default=None
        If ``self.sv_tolerance=None``, selects the number of SVD
        components to keep for each view. If none, another selection
        method is used.
    fraction_var : float, default=None
        If ``self.sv_tolerance=None``, and ``self.n_components=None``,
        selects the number of SVD components to keep for each view by
        capturing enough of the variance. If none, another selection
        method is used.
    sv_tolerance : float, optional, default=None
        Selects the number of SVD components to keep for each view by
        thresholding singular values. If none, another selection
        method is used.
    n_elbows : int, optional, default: 2
        If ``self.fraction_var=None``, ``self.sv_tolerance=None``, and
        ``self.n_components=None``, then compute the optimal embedding
        dimension using :func:`.utils.select_dimension`.
        Otherwise, ignored.
    tall : boolean, default=False
        Set to true if n_samples > n_features, speeds up SVD
    max_rank : boolean, default=False
        If true, sets the rank of the common latent space as the maximum rank
        of the individual spaces. If false, uses the minimum individual rank.
    n_jobs : int (positive), default=None
        The number of jobs to run in parallel when computing the SVDs for each
        view in `fit` and `partial_fit`. `None` means 1 job, `-1` means using
        all processors.

    Attributes
    ----------
    projection_mats_ : list of arrays
        A projection matrix for each view, from the given space to the
        latent space

    ranks_ : list of ints
        Number of left singular vectors kept for each view during the first
        SVD

    Notes
    -----
    Consider two views :math:`X_1` and :math:`X_2`. Canonical Correlation
    Analysis seeks to find vectors :math:`a_1` and :math:`a_2` to maximize
    the correlation :math:`X_1 a_1` and :math:`X_2 a_2`, expanded below.

    .. math::
        \left(\frac{a_1^TC_{12}a_2}
            {\sqrt{a_1^TC_{11}a_1a_2^TC_{22}a_2}}
            \right)

    where :math:`C_{11}`, :math:`C_{22}`, and :math:`C_{12}` are respectively
    the view 1, view 2, and between view covariance matrix estimates. GCCA
    maximizes the sum of these correlations across all pairwise views and
    computes a set of linearly independent components. This specific algorithm
    first applies principal component analysis (PCA) independently to each view
    and then aligns the most informative projections to find correlated and
    informative subspaces. Parameters that control the embedding dimension
    apply to the PCA step. The dimension of each aligned subspace is the
    maximum or minimum of the individual dimensions, per the `max_ranks`
    parameter. Using the maximum will capture the most information from all
    views but also noise from some views. Using the minimum will better remove
    noise dimensions but at the cost of information from some views.

    References
    ----------
    .. [#1GCCA] B. Afshin-Pour, G.A. Hossein-Zadeh, S.C. Strother, H.
            Soltanian-Zadeh. Enhancing reproducibility of fMRI statistical
            maps using generalized canonical correlation analysis in NPAIRS
            framework. Neuroimage, 60 (2012), pp. 1970-1981

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.embed import GCCA
    >>> # Load full dataset, labels not needed
    >>> Xs, _ = load_UCImultifeature()
    >>> gcca = GCCA(fraction_var = 0.9)
    >>> # Transform the first 5 views
    >>> Xs_latents = gcca.fit_transform(Xs[:5])
    >>> print([X.shape[1] for X in Xs_latents])
    [9, 9, 9, 9, 9]
    """

    def __init__(
            self,
            n_components=None,
            fraction_var=None,
            sv_tolerance=None,
            n_elbows=2,
            tall=False,
            max_rank=False,
            n_jobs=1,
            ):

        self.n_components = n_components
        self.fraction_var = fraction_var
        self.sv_tolerance = sv_tolerance
        self.n_elbows = n_elbows
        self.tall = tall
        self.projection_mats_ = None
        self.ranks_ = None
        self.max_rank = max_rank
        self.n_jobs = n_jobs

    def fit(self, Xs, y=None):
        r"""
        Calculates a projection from each view to a latent space such that
        the sum of pairwise latent space correlations is maximized. Each view
        'X' is normalized and the left singular vectors of 'X^T X' are
        calculated using SVD. The number of singular vectors kept is determined
        by either the percent variance explained, a given rank threshold, or a
        given number of components. The singular vectors kept are concatenated
        and SVD of that is taken and used to calculated projections for each
        view.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each view will receive its own embedding.
        y : Ignored (default = None)
            For compliance with base class

        Returns
        -------
        self : returns an instance of self.
        """

        Xs = check_Xs(Xs, multiview=True)
        n = Xs[0].shape[0]
        min_m = min(X.shape[1] for X in Xs)

        # Parallel center
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(center)(X) for X in Xs
            )
        # Parallel SVDs
        usvr = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_view)(X, n, min_m) for X in Xs
            )
        # Reshape from parallel output
        self._Uall, self._Sall, self._Vall, self.ranks_ = zip(*usvr)

        self = self._fit_multistep()

        return self

    def partial_fit(self, Xs, start=False, multiview_step=True):
        r"""
        Performs like `fit`, but will not overwrite previously fitted single
        views and instead uses them as well as the new data. Useful if the data
        needs to be processe in batches.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each view will receive its own embedding.
        start : boolean (default = False)
            If True, overwrites all prior computations.
        multiview_step : boolean, (default = True)
            If True, performs the joint SVD step on the results from individual
            views. Must be set to True in the final call.

        Returns
        -------
        self : returns an instance of self.
        """
        if not hasattr(self, '_Uall') or start:
            self._Uall = []
            self._Sall = []
            self._Vall = []
            self.ranks_ = []

        Xs = check_Xs(Xs, multiview=False)
        n = Xs[0].shape[0]
        min_m = min(X.shape[1] for X in Xs)

        # Parallel center
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(center)(X) for X in Xs
            )
        # Parallel SVDs
        usvr = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_view)(X, n, min_m) for X in Xs
            )
        # Reshape and concatenate from parallel output
        u, s, v, r = zip(*usvr)
        self._Uall += u
        self._Sall += s
        self._Vall += v
        self.ranks_ += r

        if multiview_step:
            if len(self.ranks_) < 2:
                msg = "Fewer than two single views fitted. Unable to perform \
                    multiview step."
                warnings.warn(msg, UserWarning)
            else:
                self = self._fit_multistep()

        return self

    def _fit_view(self, X, n, min_m):
        """
        Helper function to compute SVD on each view.
        """
        # Preprocess
        X[np.isnan(X)] = 0

        # compute the SVD of the data
        if self.tall:
            v, s, ut = linalg.svd(X.T, full_matrices=False)
        else:
            u, s, vt = linalg.svd(X, full_matrices=False)
            ut = u.T
            v = vt.T

        # Dimensions to reduce to
        if self.sv_tolerance:
            if not isinstance(self.sv_tolerance, float) and not isinstance(
                self.sv_tolerance, int
            ):
                raise TypeError("sv_tolerance must be numeric")
            elif self.sv_tolerance <= 0:
                raise ValueError(
                    "sv_tolerance must be greater than 0"
                    )

            rank = sum(s > self.sv_tolerance)
        elif self.n_components:
            if not isinstance(self.n_components, int):
                raise TypeError("n_components must be an integer")
            elif self.n_components <= 0:
                raise ValueError(
                    "n_components must be greater than 0"
                    )
            elif self.n_components > min((n, min_m)):
                raise ValueError(
                    "n_components must be less than or equal to the \
                        minimum input rank"
                )

            rank = self.n_components
        elif self.fraction_var:
            if not isinstance(self.fraction_var, float) and not isinstance(
                self.fraction_var, int
            ):
                raise TypeError(
                    "fraction_var must be an integer or float"
                    )
            elif self.fraction_var <= 0 or self.fraction_var > 1:
                raise ValueError("fraction_var must be in (0,1]")

            s2 = np.square(s)
            rank = sum(np.cumsum(s2 / sum(s2)) < self.fraction_var) + 1
        else:
            # Sweep over only first log2, else too large elbows
            s = s[: int(np.ceil(np.log2(np.min(X.shape))))]
            elbows, _ = select_dimension(
                s, n_elbows=self.n_elbows, threshold=None
            )
            rank = elbows[-1]

        u = ut.T[:, :rank]

        return u, s, v, rank

    def _fit_multistep(self):
        """
        Helper function to compute the SVD on the results from individuals
        view SVDs.
        """
        if self.max_rank:
            d = max(self.ranks_)
        else:
            d = min(self.ranks_)

        # Create a concatenated view of Us
        Uall_c = np.concatenate(self._Uall, axis=1)

        _, _, VV = svds(Uall_c, d)
        VV = np.flip(VV.T, axis=1)
        VV = VV[:, : min([d, VV.shape[1]])]

        # SVDS the concatenated Us
        idx_end = 0
        projection_mats = []
        n = len(self.ranks_)
        for i in range(n):
            idx_start = idx_end
            idx_end = idx_start + self.ranks_[i]
            VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)

            # Compute the canonical projections
            A = np.sqrt(n - 1) * self._Vall[i][:, : self.ranks_[i]]
            A = A @ (linalg.solve(
                np.diag(self._Sall[i][: self.ranks_[i]]), VVi
                ))

            projection_mats.append(A)

        self.projection_mats_ = projection_mats

        return self

    def transform(self, Xs, view_idx=None):
        r"""
        Embeds data matrix(s) using the fitted projection matrices. May be
        used for out-of-sample embeddings.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            A list of data matrices from each view to transform based on the
            prior fit function. If view_idx is defined, then Xs is a 2D data
            matrix corresponding to a single view.
        view_idx : int, default=None
            For transformation of a single view. If not None, then Xs is 2D
            and views_idx specifies the index of the view from which Xs comes
            from.

        Returns
        -------
        Xs_transformed : list of array-likes or array-like
            Same shape as Xs

        """
        if self.projection_mats_ is None:
            raise RuntimeError("Must call fit function before transform")
        Xs = check_Xs(Xs)
        if view_idx is not None:
            return center(Xs[0]) @ self.projection_mats_[view_idx]
        else:
            return np.array(
                [
                    center(X) @ proj
                    for X, proj in zip(Xs, self.projection_mats_)
                ]
            )

    def fit_transform(self, Xs):
        r"""
        Fits transformer to Xs and returns a transformed version of the Xs.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each view will receive its own
            transformation matrix and projection.
        Returns
        -------
        Xs_transformed : array-like, 2D if view_idx not None, otherwise
            (n_views, n_samples, self.n_components)
        """

        return self.fit(Xs).transform(Xs)
