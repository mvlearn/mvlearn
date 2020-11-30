# MIT License

# Copyright (c) [2017] [Iain Carmichael]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from copy import deepcopy
import warnings
from sklearn.utils.validation import check_is_fitted
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from .base import BaseDecomposer
from ..utils.utils import check_Xs
from ..embed.utils import select_dimension
from joblib import Parallel, delayed


class AJIVE(BaseDecomposer):
    r"""
    An implementation of Angle-based Joint and Individual Variation Explained
    [#1ajive]_. This algorithm takes multiple views and decomposes them into 3
    distinct matrices representing:
        - Low rank approximation of joint variation between views
        - Low rank approximation of individual variation within each view
        - Residual noise
    AJIVE can handle any number of views, not just two.

    Parameters
    ----------
    init_signal_ranks : list, default = None
        Initial guesses as to the rank of each view's signal.

    joint_rank : int, default = None
        Rank of the joint variation matrix. If None, will estimate the
        joint rank.

    individual_ranks : list, default = None
        Ranks of individual variation matrices. If None, will estimate the
        individual ranks. Otherwise, will use provided individual ranks.

    n_elbows : int, optional, default = 2
        If `init_signal_ranks=None`, then computes the initial signal rank
        guess using :func:`mvlearn.embed.utils.select_dimension` with
        n_elbows for each view.

    reconsider_joint_components : boolean, default = True
        Triggers `_reconsider_joint_components` function to run and removes
        columns of joint scores according to identifiability constraint.

    wedin_percentile : int, default = 5
        Percentile used for wedin (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_wedin_samples : int, default = 1000
        Number of wedin bound samples to draw.

    randdir_percentile : int, default = 95
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_randdir_samples : int, default = 1000
        Number of random direction samples to draw.

    verbose : boolean, default = False
        Prints information during runtime if True.

    n_jobs : int (positive) or None, default=None
        The number of jobs to run in parallel. `None` will run 1 job, `-1`
        uses all processors.

    random_state : int, RandomState instance or None, default=None
        Used to seed a random initialization for reproducible results.
        If None, a random initialization is used.

    Attributes
    ----------

    means_ : numpy.ndarray
        The means of each view.

    sv_thresholds_ : list of floats
        The singular value thresholds for each view based on
        initial SVD. Used to estimate the individual ranks.

    all_joint_svals_ : list of floats
        All singular values from the concatenated joint matrix.

    random_sv_samples_ : list of floats
        Random singular value samples from random direction bound.

    rand_cutoff_ : float
        Singular value squared cutoff for the random direction bound.

    wedin_samples_ : list of numpy.ndarray
        The wedin samples for each view.

    wedin_cutoff_ : float
        Singular value squared cutoff for the wedin bound.

    svalsq_cutoff_ : float
        max(rand_cutoff_, wedin_cutoff_)

    joint_rank_wedin_est_ : int
        The joint rank estimated using the wedin/random direction bound

    init_signal_ranks_ : list of ints
        Provided or estimated init_signal_ranks

    joint_rank_ : int
        The rank of the joint matrix

    joints_scores_ : numpy.ndarray
        Left singular vectors of the joint matrix

    individual_ranks_ : list of ints
        Ranks of the individual matrices for each view.

    individual_scores_ : list of numpy.ndarrays
        Left singular vectors of each view after joint matrix is removed

    individual_svals_ = list of numpy.ndarrays
        Singular values of each view after joint matrix is removed

    individual_loadings_ : list of numpy.ndarrays
        Right singular vectors of each view after joint matrix is removed

    individual_mats_ : list of numpy.ndarrays
        Individual matrices for each view, reconstructed from the SVD

    Notes
    -----

    Angle-Based Joint and Individual Variation Explained (AJIVE) is a specfic
    variation of the Joint and Individual Variation Explained (JIVE) algorithm.
    This algorithm takes :math:`k` different views with :math:`n` observations
    and :math:`d` variables and finds a basis that represents the joint
    variation and :math:`k` bases with their own ranks representing the
    individual variation of each view. Each of these individual bases is
    orthonormal to the joint basis. These bases are then used to create the
    following :math:`k` statements:

        .. math::
            X^{(i)}= J^{(i)} + I^{(i)} + E^{(i)}

    where :math:`X^{(i)}` represents the i-th view of data and :math:`J^{(i)}`,
    :math:`I^{(i)}`, and :math:`E^{(i)}` represent its joint, individual, and
    noise signal estimates respectively.

    The AJIVE algorithm calculations can be split into three seperate steps:
        - Signal Space Initial Extraction
        - Score Space Segmentation
        - Final Decomposition and Outputs

    In the **Signal Space Initial Extraction** step we compute a rank
    :math:`r_{initial}^{(i)}` singular value decomposition for each
    :math:`X^{(i)}`, the value of :math:`r_{initial}^{(i)}` can be found by
    looking at the scree plots of each view or thresholding based on singular
    value. From each singular value decomposition, the first
    :math:`r_{initial}^{(i)}` columns of of the scores matrix
    (:math:`U^{(i)}`) are taken to form :math:`\widetilde{U}^{(i)}`.

    After this, the **Score Space Segmentation** step concatenates the
    *k* :math:`\widetilde{U}^{(i)}` matrices found in the first step as
    follows:

        .. math::
            M = [\widetilde{U}^{(1)}, \dots, \widetilde{U}^{(k)}]

    From here, the :math:`r_{joint}` singular value decomposition is taken.
    :math:`r_{joint}` is estimated individually or using the wedin bound which
    quantifies how the theoretical singular subspaces are affected by noise as
    thereby quantifying the distance between rank of the original input and the
    estimation. For the scores of the singular value decomposition of
    :math:`M`, the first :math:`r_{joint}` columns are taken to obtain the
    basis, :math:`U_{joint}`. The :math:`J^{(i)}` matrix
    (joint signal estimate) can be found by projecting :math:`X^{(i)}` onto
    :math:`U_{joint}`.

    In the *Final Decomposition and Outputs* step, we project each
    :math:`X^{i}` matrix onto the orthogonal complement of :math:`U_{joint}`:

        .. math::
            X^{(i), orthog} = (I - U_{joint}U_{joint}^T)X^{(i)}


    :math:`I` in the above equation represents the identity matrix.
    From here, the :math:`I^{(i)}` matrix (individual signal estimate) can be
    found by performing the rank :math:`r_{individual}^{(i)}` singular value
    decomposition of :math:`X^{(i), orthog}`. :math:`r_{individual}^{(i)}` can
    be found by using the aforementioned singular value thresholding method.

    Finally, we can solve for the noise estimates, :math:`E^{(i)}` by using
    the equation:

        .. math::
            E^{(i)}= X^{(i)} - (J^{(i)} + I^{(i)})

    Much of this implementation has been adapted from **Iain Carmichael**
    Ph.D.'s pip-installable package, *jive*, the code for which is
    `linked here <https://github.com/idc9/py_jive>`_.


    Examples
    --------
    >>> from mvlearn.factorization.ajive import AJIVE
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> Xs, _ = load_UCImultifeature()
    >>> print(len(Xs)) # number of views
    6
    >>> print(Xs[0].shape, Xs[1].shape) # number of samples in each view
    (2000, 76) (2000, 216)
    >>> ajive = AJIVE()
    >>> Xs_transformed = ajive.fit_transform(Xs)
    >>> print(Xs_transformed[0].shape)
    (2000, 76)

    References
    ----------
    .. [#1ajive] Feng, Qing, et al. “Angle-Based Joint and Individual
            Variation Explained.” Journal of Multivariate Analysis,
            vol. 166, 2018, pp. 241–265., doi:10.1016/j.jmva.2018.03.008.

    """

    def __init__(self,
                 init_signal_ranks=None,
                 joint_rank=None,
                 individual_ranks=None,
                 n_elbows=2,
                 reconsider_joint_components=True,
                 wedin_percentile=5,
                 n_wedin_samples=1000,
                 randdir_percentile=95,
                 n_randdir_samples=1000,
                 verbose=False,
                 n_jobs=None,
                 random_state=None,
                 ):

        self.init_signal_ranks = init_signal_ranks
        self.joint_rank = joint_rank
        self.individual_ranks = individual_ranks
        self.n_elbows = n_elbows
        self.wedin_percentile = wedin_percentile
        self.n_wedin_samples = n_wedin_samples
        self.randdir_percentile = randdir_percentile
        self.n_randdir_samples = n_randdir_samples
        self.reconsider_joint_components = reconsider_joint_components
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._check_params()

    def fit(self, Xs, y=None):
        r"""
        Learns a decomposition of the views into joint and individual
        matrices.

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
        # Check data
        Xs, n_views, n_samples, n_features = check_Xs(
            Xs, return_dimensions=True
        )
        self.view_shapes_ = [(n_samples, f) for f in n_features]

        # Check parameters with data
        self._check_fit_params(n_views, n_samples, n_features)

        # Estimate signal ranks if not given
        if self.init_signal_ranks is None:
            self.init_signal_ranks_ = []
            for X in Xs:
                elbows, _ = select_dimension(X, n_elbows=self.n_elbows)
                self.init_signal_ranks_.append(elbows[-1])
        else:
            self.init_signal_ranks_ = self.init_signal_ranks

        # Center columns and store
        self.means_ = [np.mean(X, axis=0) for X in Xs]
        Xs = [X - mean for X, mean in zip(Xs, self.means_)]

        # SVD to extract signal on each view
        self.sv_thresholds_ = []
        score_matrices = []
        sval_matrices = []
        loading_matrices = []
        for i, (X, signal_rank) in enumerate(
            zip(Xs, self.init_signal_ranks_)
        ):
            if signal_rank >= min(X.shape):
                warnings.warn(f"Given rank {signal_rank} greater or equal to \
                    maximum possible full rank {min(X.shape)}. Using \
                    1 minus full instead", RuntimeWarning)
                signal_rank = min(X.shape) - 1
                self.init_signal_ranks_[i] = signal_rank
            # signal rank + 1 to get individual rank sv threshold
            U, D, V = _svd_wrapper(X, signal_rank + 1)

            # The SV threshold is halfway between the signal_rank
            # and signal_rank + 1 singular value.
            self.sv_thresholds_.append(
                (D[signal_rank - 1] + D[signal_rank])/2
                )

            # Store SVD results
            score_matrices.append(U[:, :signal_rank])
            sval_matrices.append(D[:signal_rank])
            loading_matrices.append(V[:, :signal_rank])

        # SVD of joint signal matrix. Here we are trying to estimate joint
        # rank and find an apt joint basis.
        joint_scores_matrix = np.hstack(score_matrices)
        joint_scores, joint_svals, joint_loadings = \
            _svd_wrapper(joint_scores_matrix)
        self.all_joint_svals_ = deepcopy(joint_svals)

        # estimate joint rank using wedin bound and random direction if a
        # joint rank estimate has not already been provided
        if self.joint_rank is None:
            # Calculate sv samples
            np.random.seed(self.random_state)
            self.random_sv_samples_ = \
                _sample_randdir(n_samples, self.init_signal_ranks_,
                                self.n_randdir_samples, self.n_jobs)

            # Compute wedin samples
            np.random.seed(self.random_state)
            self.wedin_samples_ = [
                _get_wedin_samples(
                    X, U, D, V, rank, self.n_wedin_samples, self.n_jobs
                    )
                for X, U, D, V, rank in zip(
                    Xs, score_matrices, sval_matrices, loading_matrices,
                    self.init_signal_ranks_)
                ]

            # Joint singular value lower bounds (Lemma 3)
            self.wedin_sv_samples_ = n_views - \
                np.array([sum([w[i]**2 for w in self.wedin_samples_])
                         for i in range(self.n_wedin_samples)])

            # Now calculate joint matrix rank

            self.wedin_cutoff_ = np.percentile(self.wedin_sv_samples_,
                                               self.wedin_percentile)
            self.rand_cutoff_ = np.percentile(self.random_sv_samples_,
                                              self.randdir_percentile)
            self.svalsq_cutoff_ = max(self.wedin_cutoff_, self.rand_cutoff_)
            self.joint_rank_wedin_est_ = sum(joint_svals ** 2 >
                                             self.svalsq_cutoff_)
            self.joint_rank_ = deepcopy(self.joint_rank_wedin_est_)
        else:
            self.joint_rank_ = deepcopy(self.joint_rank)

        # check identifiability constraint
        if self.reconsider_joint_components:
            self.joint_scores_, _, _, self.joint_rank_ = \
                _reconsider_joint_components(Xs, self.sv_thresholds_,
                                             joint_scores, joint_svals,
                                             joint_loadings, self.joint_rank_,
                                             self.verbose)
        # Joint basis
        self.joint_scores_ = self.joint_scores_[:, :self.joint_rank_]

        # view estimates
        self.individual_scores_ = []
        self.individual_svals_ = []
        self.individual_loadings_ = []
        if self.individual_ranks is None:
            self.individual_ranks_ = []
        else:
            self.individual_ranks_ = self.individual_ranks

        for i, (X, sv_threshold) in enumerate(zip(Xs, self.sv_thresholds_)):
            # View specific joint space creation
            # projecting X onto the joint space then compute SVD
            # Then find the orthogonal complement to the joint matrix
            if self.joint_rank_ != 0:
                J = np.array(np.dot(self.joint_scores_,
                                    np.dot(self.joint_scores_.T, X)))
                U, D, V = _svd_wrapper(J, self.joint_rank_)
                X_orthog = X - J
            else:
                U, D, V = None, None, None
                X_orthog = X

            # estimate individual rank using sv threshold, then compute SVD
            if self.individual_ranks is None:
                max_rank = min(X.shape) - self.joint_rank_
                if max_rank == 0:
                    rank = 0
                else:
                    U, D, V = _svd_wrapper(X_orthog, max_rank)
                    rank = sum(D > sv_threshold)

                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U = U[:, :rank]
                    D = D[:rank]
                    V = V[:, :rank]

                self.individual_ranks_.append(rank)
            else:  # if user inputs rank list for individual matrices
                rank = self.individual_ranks_[i]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = _svd_wrapper(X_orthog, rank)

            self.individual_scores_.append(U)
            self.individual_svals_.append(D)
            self.individual_loadings_.append(V)

        return self

    def transform(self, Xs):
        r"""

        Returns the joint matrices from each view.

        Parameters
        ----------

        Xs: list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Data to be transformed

        Returns
        -------

        Js : list of numpy.ndarray
            Joint matrices of each inputted view.

        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)

        if self.joint_rank_ == 0:
            Js = [np.zeros(s) for s in self.view_shapes_]
            warnings.warn(
                "Joint rank is 0, returning zero matrix.", RuntimeWarning
                )
        else:
            Js = [self.joint_scores_ @ self.joint_scores_.T @ (X - mean)
                  for X, mean in zip(Xs, self.means_)]
        return Js

    def inverse_transform(self, Xs_transformed):
        r"""Recover original data from transformed data.

        Parameters
        ----------
        Xs_transformed: list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Estimated joint views

        Returns
        -------
        Xs : list of arrays
            The summed individual and joint blocks and mean
        """
        check_is_fitted(self)
        Xs_transformed = check_Xs(Xs_transformed)

        return [X + I + mean for X, I, mean in zip(
            Xs_transformed, self.individual_mats_, self.means_)]

    def _check_params(self):
        if self.joint_rank is not None and (
                (self.init_signal_ranks is not None and
                    self.joint_rank > sum(self.init_signal_ranks)) or
                self.joint_rank < 0):
            raise ValueError(
                "joint_rank must be between 0 and the sum of the \
                init_signal_ranks"
            )

        if self.init_signal_ranks is None and self.n_elbows is None:
            raise ValueError("Either init_signal_ranks must be provided a \
                list or n_elbows must be a positive integer")

        if not isinstance(self.n_wedin_samples, int) or \
                self.n_wedin_samples < 1:
            raise ValueError("n_wedin_samples must be a positive integer")

        if not isinstance(self.n_randdir_samples, int) or \
                self.n_randdir_samples < 1:
            raise ValueError("n_randdir_samples must be a positive integer")

        if self.init_signal_ranks is not None and \
                not isinstance(self.init_signal_ranks, (list, np.ndarray)):
            raise ValueError("init_signal_ranks must be of type list if \
                not None")

        if self.individual_ranks is not None and \
                not isinstance(self.individual_ranks, (list, np.ndarray)):
            raise ValueError("individual_ranks must be of type list if \
                not None")

    def _check_fit_params(self, n_views, n_samples, n_features):
        max_ranks = np.minimum(n_samples, n_features)

        if self.init_signal_ranks is not None and (
            not np.all(1 <= np.asarray(self.init_signal_ranks)) or not
                np.all(np.asarray(self.init_signal_ranks) <= max_ranks)):
            raise ValueError(
                "init_signal_ranks must all be between 1 and the minimum \
                of the number of rows and columns for each view")

        if self.individual_ranks is not None and \
                len(self.individual_ranks) != n_views:
            raise ValueError("individual_ranks must be of length \
                n_views")

        if self.init_signal_ranks is not None and \
                len(self.init_signal_ranks) != n_views:
            raise ValueError("init_signal_ranks must be of length \
                n_views")

    @property
    def individual_mats_(self):
        """Computes full individual matrices from saved decompositions"""
        check_is_fitted(self)
        Is = []
        for i, r in enumerate(self.individual_ranks_):
            if r == 0:
                Is.append(np.zeros(self.view_shapes_[i]))
            else:
                Is.append(
                    self.individual_scores_[i] @
                    np.diag(self.individual_svals_[i]) @
                    self.individual_loadings_[i].T
                )
        return Is


def _reconsider_joint_components(
    Xs, sv_thresholds, joint_scores, joint_svals, joint_loadings, joint_rank,
    verbose
):
    """
    Checks the identifiability constraint on the joint singular values and
    removes columns that fail. Set `verbose=True` to print removed columns.
    """

    # check identifiability constraint
    to_keep = set(range(joint_rank))
    for X, sv_threshold in zip(Xs, sv_thresholds):
        for j in to_keep:
            # This might be joint_sv
            score = X.T @ joint_scores[:, j]
            sv = np.linalg.norm(score)

            # if sv is below the threshold for any data block remove j
            if sv < sv_threshold:
                if verbose:
                    print(f"Excluding column {j}, below identifiability \
                        threshold")
                to_keep.remove(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    joint_scores = joint_scores[:, list(to_keep)]
    joint_loadings = joint_loadings[:, list(to_keep)]
    joint_svals = joint_svals[list(to_keep)]

    return joint_scores, joint_svals, joint_loadings, joint_rank


def _sample_randdir(num_obs, signal_ranks, R=1000, n_jobs=None):
    r"""
    Draws samples for the random direction bound.

    Parameters
    ----------

    num_obs: int
        Number of observations.

    signal_ranks: list of ints
        The initial signal ranks for each block.

    R: int
        Number of samples to draw.

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    Returns
    -------
    random_sv_samples: numpy.ndarray, shape (R,)
    """
    random_sv_samples = Parallel(n_jobs=n_jobs)(
        delayed(_get_randdir_sample)(num_obs, signal_ranks) for i in range(R)
        )

    return np.array(random_sv_samples)


def _get_randdir_sample(num_obs, signal_ranks):
    """
    Computes squared largest singular value of random joint matrix
    """
    M = [None for _ in range(len(signal_ranks))]
    for k in range(len(signal_ranks)):
        # sample random orthonormal basis
        Z = np.random.normal(size=(num_obs, signal_ranks[k]))
        M[k] = np.linalg.qr(Z)[0]

    M = np.bmat(M)
    _, svs, __ = _svd_wrapper(M, rank=1)

    return max(svs) ** 2


def _get_wedin_samples(X, U, D, V, rank, R=1000, n_jobs=None):
    r"""
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data

    U, D, V : array-likes
        The partial SVD of X=UDV^T

    rank : int
        The rank of the signal space

    R : int
        Number of samples for resampling procedure

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    Returns
    -------
    wedin_bound_samples : list of resampled wedin bounds
    """

    # resample for U and V
    U_norm_samples = _norms_sample_project(
        X.T, U[:, :rank], R, n_jobs
    )

    V_norm_samples = _norms_sample_project(
        X, V[:, :rank], R, n_jobs
    )

    sigma_min = D[rank - 1]
    wedin_bound_samples = [
        min(max(U_norm_samples[r], V_norm_samples[r]) / sigma_min, 1)
        for r in range(R)
    ]

    return wedin_bound_samples


def _norms_sample_project(X, basis, R=1000, n_jobs=None):
    r"""
    Samples vectors from space orthogonal to signal space as follows
    - sample random vector from isotropic distribution
    - project onto orthogonal complement of signal space and normalize

    Parameters
    ---------
    X: array-like, shape (N, D)
        The observed data

    B: array-like
        The basis for the signal col/rows space (e.g. the left/right singular\
        vectors)

    rank: int
        Number of columns to resample

    R: int
        Number of samples

    n_jobs: int, None
        Number of jobs for parallel processing.

    Returns
    -------
    samples : list of the resampled noise norms
    """

    samples = Parallel(n_jobs=n_jobs)(
        delayed(_get_noise_sample)(X, basis) for i in range(R)
        )

    return np.array(samples)


def _get_noise_sample(X, basis):
    """
    Estimates magnitude of noise matrix projected onto signal matrix.
    """
    # sample from isotropic distribution
    vecs = np.random.normal(size=basis.shape)

    # project onto space orthogonal to cols of B
    # vecs = (np.eye(dim) - np.dot(basis, basis.T)).dot(vecs)
    vecs = vecs - np.dot(basis, np.dot(basis.T, vecs))

    # orthonormalize
    vecs, _ = np.linalg.qr(vecs)

    # compute operator L2 norm
    return np.linalg.norm(X.dot(vecs), ord=2)


def _svd_wrapper(X, rank=None):
    r"""
    Computes the full or partial SVD of a matrix. Handles the case where
    X is either dense or sparse.

    Parameters
    ----------
    X : array-like, shape (N, D)

    rank : int
        rank of the desired SVD. If `None`, the full SVD is used.

    Returns
    -------
    U : array-like, shape (N, rank)
        Orthonormal matrix of left singular vectors.

    D : list, shape (rank,)
        Singular values in decreasing order

    V : array-like, shape (D, rank)
        Orthonormal matrix of right singular vectors

    """
    full = False
    if rank is None or rank == min(X.shape):
        full = True

    if not full:
        assert rank <= min(X.shape) - 1  # svds cannot compute the full svd
        U, D, V = svds(X, rank)

        # Sort in decreasing order
        sv_reordering = np.argsort(-D)
        U = U[:, sv_reordering]
        D = D[sv_reordering]
        V = V.T[:, sv_reordering]

    else:
        U, D, V = svd(X, full_matrices=False)

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V.T[:, :rank]

    return U, D, V
