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
from scipy.sparse import issparse
from copy import deepcopy
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from .base import BaseDecomposer
from ..utils.utils import check_Xs
from ..embed.utils import select_dimension
from .ajive_utils.block_visualization import _data_block_heatmaps#, _ajive_full_estimate_heatmaps
from .ajive_utils.utils import _svd_wrapper, _centering
from .ajive_utils.wedin_bound import _get_wedin_samples
from .ajive_utils.random_direction import sample_randdir


class AJIVE(BaseDecomposer):
    r"""
    An implementation of Angle-based Joint and Individual Variation Explained
    [#1ajive]_. This algorithm takes multiple views and decomposes them into 3
    distinct matrices representing:
        - Low rank approximation of individual variation within each view
        - Low rank approximation of joint variation between views
        - Residual noise
    AJIVE can handle any number of views, not just two.

    Parameters
    ----------
    init_signal_ranks : list, default = None
        Initial guesses as to the rank of each view's signal.

    joint_rank : int, default = None
        Rank of the joint variation matrix. If None, will estimate the
        joint rank. Otherwise, will use provided joint rank.

    indiv_ranks : list, default = None
        Ranks of individual variation matrices. If None, will estimate the
        individual ranks. Otherwise, will use provided individual ranks.

    n_elbows : int, optional, default: 2
        If ``init_signal_ranks=None``, then computes the initial signal rank
        guess using :func:`mvlearn.embed.utils.select_dimension` with
        n_elbows for each view.

    center : bool, default = True
        Boolean for centering matrices.

    reconsider_joint_components : bool, default = True
        Triggers _reconsider_joint_components function to run and removes
        columns of joint scores according to identifiability constraint.

    wedin_percentile : int, default = 5
        Percentile used for wedin (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_wedin_samples : int, default = 1000
        Number of wedin bound samples to draw.

    precomp_wedin_samples : list of array-like
        Wedin samples that are precomputed for each view.

    randdir_percentile : int, default = 95
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_randdir_samples : int, default = 1000
        Number of random direction samples to draw.

    precomp_randdir_samples : array-like, default = None
        Precomputed random direction samples.

    Attributes
    ----------

    common_ : The class mvlearn.factorization.ajive_utils.pca.pca
        The common joint space found using pca class in same directory

    blocks_ : dict
        The block-specific results.

    centers_ : dict
        The the centering vectors computed for each matrix.

    sv_thresholds_ : dict
        The singular value thresholds computed for each block based on
        initial SVD. Eventually used to estimate the individual ranks.

    all_joint_svals_ : list
        All singular values from the concatenated joint matrix.

    random_sv_samples_ : list
        Random singular value samples from random direction bound.

    rand_cutoff_ : float
        Singular value squared cutoff for the random direction bound.

    wedin_samples_ : dict
        The wedin samples for each view.

    wedin_cutoff_ : float
        Singular value squared cutoff for the wedin bound.

    svalsq_cutoff_ : float
        max(rand_cutoff_, wedin_cutoff_)

    joint_rank_wedin_est_ : int
        The joint rank estimated using the wedin/random direction bound.

    init_signal_rank_ : dict
        init_signal_rank in a dictionary of items for each view.

    joint_rank_ : int
        The rank of the joint matrix

    indiv_ranks_ : dict of ints
        Ranks of the individual matrices for each view.

    center_ : dict
        Center in a dict of items for each view.

    is_fit_ : bool, default = False
        Returns whether data has been fit yet

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
    >>> print(len(Xs)) # number of samples in each view
    6
    >>> print(Xs[0].shape, Xs[1].shape) # number of samples in each view
    (2000, 76) (2000, 216)
    >>> ajive = AJIVE(init_signal_ranks=[2,2])
    >>> b = ajive.fit(Xs).predict()
    >>> print(b)
    6
    >>> print(b[0][0].shape,b[1][0].shape)  # (V1 joint mat, V2 joint mat)
    (2000, 76) (2000, 216)
    >>> print(b[0][1].shape,b[1][1].shape)  # (V1 indiv mat, V2 indiv mat)
    (2000, 76) (2000, 216)
    >>> print(b[0][2].shape,b[1][2].shape)  # (V1 noise mat, V2 noise mat)
    (2000, 76) (2000, 216)

    References
    ----------
    .. [#1ajive] Feng, Qing, et al. “Angle-Based Joint and Individual
            Variation Explained.” Journal of Multivariate Analysis,
            vol. 166, 2018, pp. 241–265., doi:10.1016/j.jmva.2018.03.008.

    """

    def __init__(self,
                 init_signal_ranks=None,
                 joint_rank=None,
                 indiv_ranks=None,
                 n_elbows=None,
                 center=True,
                 reconsider_joint_components=True,
                 wedin_percentile=5,
                 n_wedin_samples=1000,
                 precomp_wedin_samples=None,
                 randdir_percentile=95,
                 n_randdir_samples=1000,
                 precomp_randdir_samples=None,
                 store_full=True
                 ):

        self.init_signal_ranks = init_signal_ranks
        self.joint_rank = joint_rank
        self.indiv_ranks = indiv_ranks
        self.n_elbows = n_elbows
        self.center = center

        self.wedin_percentile = wedin_percentile
        self.precomp_wedin_samples = precomp_wedin_samples
        if precomp_wedin_samples is not None:
            self.n_wedin_samples = len(precomp_wedin_samples[0])
        else:
            self.n_wedin_samples = n_wedin_samples

        self.randdir_percentile = randdir_percentile
        self.precomp_randdir_samples = precomp_randdir_samples
        if precomp_randdir_samples is not None:
            self.n_randdir_samples = len(precomp_randdir_samples)
        else:
            self.n_randdir_samples = n_randdir_samples

        self.reconsider_joint_components = reconsider_joint_components
        self.store_full = store_full


    def fit(self, Xs, precomp_init_svds=None):
        r"""
        Learns a decomposition of the views.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The different views that are input. Input as data matrices.

        precomp_init_svds: dict or list
            Precomputed initial SVD. Must have one entry for each view.
            The SVD should be an ordered list of 3 matrices (scores, svals,
            loadings), see output of `ajive_utils/utils/svd_wrapper`
            for formatting details.

        Returns
        -------
        self : returns the object instance.
        """
        # Check data
        Xs, n_views, n_samples, n_features = check_Xs(
            Xs, return_dimensions=True
        )

        # Check parameters
        self._check_params(n_samples, n_features)

        # Estimate signal ranks if not given
        if self.init_signal_ranks is None:
            self.init_signal_ranks_ = []
            for X in Xs:
                elbows, _ = select_dimension(X, n_elbows=self.n_elbows)
                self.init_signal_ranks_.append(elbows[-1])
        else:
            self.init_signal_ranks_ = self.init_signal_ranks
        
        # Check individual ranks
        # TODO

        # TODO Center

        # SVD to extract signal on each view
        self.sv_thresholds_ = []
        score_matrices = []
        sval_matrices = []
        loading_matrices = []
        for i, (X, signal_rank) in enumerate(zip(
                Xs, self.init_signal_ranks_
            )):
            # compute SVD with rank init_signal_rank + 1 for view
            if precomp_init_svds is None:
                # signal rank + 1 to get individual rank sv threshold
                U, D, V = _svd_wrapper(X, signal_rank + 1)
            # If precomputed return values already found
            else:
                U, D, V = precomp_init_svds[i]

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

            # Calculating sv samples if not provided
            if self.precomp_randdir_samples is None:
                init_rank_list = list(self.init_signal_ranks_)
                self.random_sv_samples_ = \
                    sample_randdir(n_views,
                                   signal_ranks=init_rank_list,
                                   R=self.n_randdir_samples)
            else:
                self.random_sv_samples_ = self.precomp_randdir_samples

            # if the wedin samples are not already provided compute them
            if self.precomp_wedin_samples is None:
                self.wedin_samples_ = [
                    _get_wedin_samples( #TODO add parallel n_jobs
                        X=X, U=U,D=D,V=V,rank=rank,R=self.n_wedin_samples
                        )
                    for X,U,D,V,rank in zip(
                        Xs, score_matrices, sval_matrices, loading_matrices,
                        self.init_signal_ranks_)
                    ]
            else:
                self.wedin_samples_ = precomp_wedin_samples

            # Joint singular value lower bounds (Lemma 3)
            self.wedin_sv_samples_ = n_views - \
                np.array([sum([w[i]**2 for w in self.wedin_samples_])
                    for i in range(self.n_wedin_samples)]
                )

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
            joint_scores, joint_svals, joint_loadings, self.joint_rank_ = \
                _reconsider_joint_components(Xs,
                                             self.sv_thresholds_,
                                             joint_scores,
                                             joint_svals,
                                             joint_loadings,
                                             self.joint_rank_)
        # Joint basis
        self.joint_scores_ = joint_scores[:, :self.joint_rank_]
        self.joint_svals_ = joint_svals[:self.joint_rank_]
        self.joint_loadings_ = joint_loadings[:, :self.joint_rank_]

        # view estimates
        self.indiv_scores_ = []
        self.indiv_svals_ = []
        self.indiv_loadings_ = []
        if self.indiv_ranks is None:
            self.indiv_ranks_ = []

        for i, (X, sv_threshold) in enumerate(zip(Xs, self.sv_thresholds_)):
            # View specific joint space creation
            # projecting X onto the joint space then compute SVD
            if self.joint_rank_ != 0:
                J = np.array(np.dot(self.joint_scores_,
                                    np.dot(self.joint_scores_.T, X)))
                U, D, V = _svd_wrapper(J, self.joint_rank_)
                if not self.store_full:
                    J = None  # kill J matrix to save memory

            else:
                U, D, V = None, None, None
                if self.store_full:
                    J = np.zeros(shape=X.shape)
                else:
                    J = None

            # Here we are creating the individual representations for
            # each view.

            # Finding the orthogonal complement to the joint matrix
            if self.joint_rank_ == 0:
                X_orthog = X
            else:
                X_orthog = X - np.dot(self.joint_scores_, np.dot(self.joint_scores_.T, X))

            # estimate individual rank using sv threshold, then compute SVD
            if self.indiv_ranks is None:
                max_rank = min(X.shape) - self.joint_rank_  # saves computation
                U, D, V = _svd_wrapper(X_orthog, max_rank)
                rank = sum(D > sv_threshold)

                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U = U[:, :rank]
                    D = D[:rank]
                    V = V[:, :rank]

                self.indiv_ranks_.append(rank)

            # SVD on the orthogonal complement
            else:  # if user inputs rank list for individual matrices
                rank = self.indiv_ranks[i]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = _svd_wrapper(X_orthog, rank)

            self.indiv_scores_.append(U)
            self.indiv_svals_.append(D)
            self.indiv_loadings_.append(V)

        return self

    def transform(self, Xs):
        r"""

        Returns the joint, individual, and noise components of each view from
        the fitted decomposition. Only works on the data inputted in `fit`.

        Parameters
        ----------

        Xs: list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Data to be transformed

        Returns
        -------

        Is : list of array-likes or numpy.ndarray
            Individual portions of each inputted view.
            TODO `multiview_output` is False

        """
        check_is_fitted(self)

        Xs = check_Xs(Xs)
        
        Js = [self.joint_scores_ @ self.joint_scores_.T @ X for X in Xs]

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
            The summed individual and joint blocks
        """
        check_is_fitted(self)

        return [X + I for X, I in zip(Xs_transformed, self.indiv_mats_)]


    def _check_params(self, n_samples, n_features):
        max_ranks = np.minimum(n_samples, n_features)

        if self.init_signal_ranks is not None and \
            not np.all(1 <= np.asarray(self.init_signal_ranks)) or \
            not np.all(self.init_signal_ranks <= max_ranks):
            raise ValueError(
                "init_signal_ranks must all be between 1 and the minimum of \
                the number of rows and columns for each view"
            )


        if self.joint_rank is not None and \
            self.joint_rank > sum(self.init_signal_ranks):
            raise ValueError(
                "joint_rank must be smaller than the sum of the \
                initial_signal_ranks"
            )


    @property
    def indiv_mats_(self):
        Is = [
            U @ np.diag(D) @ V.T for U,D,V in zip(
                self.indiv_scores_, self.indiv_svals_, self.indiv_loadings_)
        ]
        return Is
           

def data_block_heatmaps(Xs): # TODO refactor plotting utilities
    r"""
    Plots n_views heatmaps in a singular row. It is recommended to set
    a figure size as is shown in the tutorial.

    Parameters
    ----------
    Xs : dict or list of array-likes
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The different views to plot.
    """
    _data_block_heatmaps(Xs)


def ajive_full_estimate_heatmaps(Xs, full_block_estimates, names=None): # TODO refactor plotting utilities
    r"""
    Plots four heatmaps for each of the views:
        - Full initial data
        - Full joint signal estimate
        - Full individual signal estimate
        - Full noise estimate
    It is recommended to set a figure size as shown in the tutorial.

    Parameters
    ----------
    Xs: list of array-likes
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The different views that are input. Input as data matrices.

    full_block_estimates: dict
        Dict that is returned from the `ajive.predict()` function

    names: list
        The names of the views.
    """
    _ajive_full_estimate_heatmaps(Xs, full_block_estimates, names)


def _reconsider_joint_components(
    Xs, sv_thresholds, joint_scores, joint_svals, joint_loadings, joint_rank
):
    """
    Checks the identifiability constraint on the joint singular values
    """

    # check identifiability constraint
    to_keep = set(range(joint_rank))
    for X, sv_threshold in zip(Xs, sv_thresholds):
        for j in range(joint_rank):
            # This might be joint_sv
            score = X.T @ joint_scores[:, j]
            sv = np.linalg.norm(score)

            # if sv is below the threshold for any data block remove j
            if sv < sv_threshold:
                # print("removing column " + str(j)) # TODO verbose option
                to_keep.remove(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    joint_scores = joint_scores[:, list(to_keep)]
    joint_loadings = joint_loadings[:, list(to_keep)]
    joint_svals = joint_svals[list(to_keep)]
    return joint_scores, joint_svals, joint_loadings, joint_rank
