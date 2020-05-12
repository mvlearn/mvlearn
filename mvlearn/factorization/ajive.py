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

import numpy as np
from scipy.sparse import issparse
from copy import deepcopy
from sklearn.externals.joblib import load, dump
import pandas as pd
from ..utils.utils import check_Xs
from .ajive_utils.block_visualization import _data_block_heatmaps, \
    _ajive_full_estimate_heatmaps

from .ajive_utils.utils import svd_wrapper, centering
from .ajive_utils.wedin_bound import get_wedin_samples
from .ajive_utils.random_direction import sample_randdir
from .ajive_utils.pca import pca


class ajive(object):
    r"""
    An implementation of Angle-based Joint and Individual Variation Explained.
    This algorithm takes multiple input views with the same number of samples
    and decomposes them into 3 distinct matrices representing [#1ajive]_:
        - Individual variation of each particular view
        - Joint variation shared by all views
        - Noise

    Parameters
    ----------
    init_signal_ranks: list or dict
        The initial signal ranks.

    joint_rank: int, default = None
        Rank of the joint variation matrix. If None, will estimate the
        joint rank. Otherwise, will use provided joint rank.

    indiv_ranks: list or dict
        Ranks of individual variation matrices. If None, will estimate the
        individual ranks. Otherwise, will use provided individual ranks.

    center: bool, default = True
        Boolean for centering matrices.

    reconsider_joint_components: bool, default = True
        Triggers _reconsider_joint_components function to run and removes
        columns of joint scores according to identifiability constraint.

    wedin_percentile: int, default = 5
        Percentile used for wedin (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_wedin_samples: int, default = 1000
        Number of wedin bound samples to draw.

    precomp_wedin_samples: Dict of array-like or list of array-like
        Wedin samples that are precomputed for each view.

    randdir_percentile: int, default = 95
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_randdir_samples: int, default = 1000
        Number of random direction samples to draw.

    precomp_randdir_samples: array-like, default = None
        Precomputed random direction samples.

    n_jobs: int, default = None
        Number of jobs for parallel processing wedin samples and random
        direction samples using sklearn.externals.joblib.Parallel.
        If None, will not use parallel processing.

    Attributes
    ----------

    common: mvlearn.factorization.pca.pca
        The common joint space found using pca class in same directory

    blocks: dict
        The block-specific results.

    centers_: dict
        The the centering vectors computed for each matrix.

    sv_threshold_: dict
        The singular value thresholds computed for each block based on
        initial SVD. Eventually used to estimate the individual ranks.

    all_joint_svals_: list
        All singular values from the concatenated joint matrix.

    random_sv_samples_: list
        Random singular value samples from random direction bound.

    rand_cutoff_: float
        Singular value squared cutoff for the random direction bound.

    wedin_samples_: dict
        The wedin samples for each view.

    wedin_cutoff_: float
        Singular value squared cutoff for the wedin bound.

    svalsq_cutoff_: float
        max(rand_cutoff_, wedin_cutoff_)

    joint_rank_wedin_est_: int
        The joint rank estimated using the wedin/random direction bound.

    joint_rank: int
        The rank of the joint matrix

    indiv_ranks: dict of ints
        Ranks of the individual matrices for each view.

    is_fit: bool, default = False
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
    decomposition of :math:`X^{(k), orthog}`. :math:`r_{individual}^{(i)}` can
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
    >>> from mvlearn.factorization.ajive import ajive
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> Xs, _ = load_UCImultifeature()
    >>> print(len(Xs)) # number of samples in each view
    6
    >>> print(Xs[0].shape, Xs[1].shape) # number of samples in each view
    (2000, 76) (2000, 216)
    >>> Ajive = ajive(init_signal_ranks=[2,2])
    >>> Ajive.fit(Xs)
    >>> b = Ajive.predict()
    >>> print(len(b.keys()))
    6
    >>> print(b[0]['joint'].shape,b[1]['joint'].shape)
    (2000, 76) (2000, 216)
    >>> print(b[0]['individual'].shape,b[1]['individual'].shape)
    (2000, 76) (2000, 216)
    >>> print(b[0]['noise'].shape,b[1]['noise'].shape)
    (2000, 76) (2000, 216)

    References
    ----------
    .. [#1ajive] Feng, Qing, et al. “Angle-Based Joint and Individual
            Variation Explained.” Journal of Multivariate Analysis,
            vol. 166, 2018, pp. 241–265., doi:10.1016/j.jmva.2018.03.008.

    """

    def __init__(self,
                 init_signal_ranks,
                 joint_rank=None, indiv_ranks=None,
                 center=True,
                 reconsider_joint_components=True,
                 wedin_percentile=5, n_wedin_samples=1000,
                 precomp_wedin_samples=None,
                 randdir_percentile=95, n_randdir_samples=1000,
                 precomp_randdir_samples=None,
                 store_full=True, n_jobs=None):

        self.init_signal_ranks = init_signal_ranks
        self.joint_rank = joint_rank
        self.indiv_ranks = indiv_ranks

        self.center = center

        self.wedin_percentile = wedin_percentile
        self.n_wedin_samples = n_wedin_samples
        self.wedin_samples_ = precomp_wedin_samples
        if precomp_wedin_samples is not None:
            self.n_wedin_samples = len(list(precomp_wedin_samples.values())[0])

        self.randdir_percentile = randdir_percentile
        self.n_randdir_samples = n_randdir_samples
        self.random_sv_samples_ = precomp_randdir_samples
        if precomp_randdir_samples is not None:
            self.n_randdir_samples = len(precomp_randdir_samples)

        self.reconsider_joint_components = reconsider_joint_components

        self.store_full = store_full

        self.n_jobs = n_jobs

    def __repr__(self):

        if self.is_fit:
            r = "joint rank: {}".format(self.common.rank)
            for bn in self.block_names:
                indiv_rank = self.blocks[bn].individual.rank
                r += ", block {} indiv rank: {}".format(bn, indiv_rank)
            return r

        else:
            return "No data has been fitted yet"

    def fit(self, blocks, precomp_init_svd=None):
        r"""
        Fits the AJIVE decomposition.

        Parameters
        ----------
        blocks: dict or list of array-likes
            - blocks length: n_views
            - blocks[i] shape: (n_samples, n_features_i)
            The different views that are input. Input as data matrices.
            If dict, will name blocks by keys, otherwise blocks are named by
            0, 1, ...K.

        precomp_init_svd: dict or list
            Precomputed initial SVD. Must have one entry for each data block.
            The SVD should be a 3 tuple (scores, svals, loadings), see output
            of .svd_wrapper for formatting details.

        """

        blocks, self.init_signal_ranks, self.indiv_ranks, precomp_init_svd,\
            self.center, obs_names, var_names, self.shapes_ = \
                            _arg_checker(blocks,
                                         self.init_signal_ranks,
                                         self.joint_rank,
                                         self.indiv_ranks,
                                         precomp_init_svd,
                                         self.center)

        block_names = list(blocks.keys())
        num_obs = list(blocks.values())[0].shape[0]  # number of views

        # centering views
        self.centers_ = {}
        for bn in block_names:
            blocks[bn], self.centers_[bn] = centering(blocks[bn],
                                                      method=self.center[bn])

        # SVD to extract signal on each view

        init_signal_svd = {}
        self.sv_threshold_ = {}
        for bn in block_names:

            # compute SVD with rank init_signal_ranks[bn] + 1 for view
            if precomp_init_svd[bn] is None:
                # signal rank + 1 to get individual rank sv threshold
                U, D, V = svd_wrapper(blocks[bn],
                                      self.init_signal_ranks[bn] + 1)
            # If precomputed return values already found
            else:
                U = precomp_init_svd[bn]["scores"]
                D = precomp_init_svd[bn]["svals"]
                V = precomp_init_svd[bn]["loadings"]

            # The SV threshold is halfway between the init_signal_ranks[bn]th
            # and init_signal_ranks[bn] + 1 st singular value.
            self.sv_threshold_[bn] = (D[self.init_signal_ranks[bn] - 1]
                                      + D[self.init_signal_ranks[bn]])/2

            init_signal_svd[bn] = {'scores':
                                   U[:, 0:self.init_signal_ranks[bn]],
                                   'svals':
                                   D[0:self.init_signal_ranks[bn]],
                                   'loadings':
                                   V[:, 0:self.init_signal_ranks[bn]]}

        # SVD of joint signal matrix. Here we are trying to estimate joint
        # rank and find an apt joint basis.

        joint_scores_matrix = \
            np.bmat([init_signal_svd[bn]['scores'] for bn in block_names])
        joint_scores, joint_svals, joint_loadings = \
            svd_wrapper(joint_scores_matrix)
        self.all_joint_svals_ = deepcopy(joint_svals)

        # estimate joint rank using wedin bound and random direction if a
        # joint rank estimate has not already been provided

        if self.joint_rank is None:

            # Calculating sv samples if not provided
            if self.random_sv_samples_ is None:
                init_rank_list = list(self.init_signal_ranks.values())
                self.random_sv_samples_ = \
                    sample_randdir(num_obs,
                                   signal_ranks= init_rank_list,
                                   R=self.n_randdir_samples,
                                   n_jobs=self.n_jobs)

            # if the wedin samples are not already provided compute them
            if self.wedin_samples_ is None:
                self.wedin_samples_ = {}
                for bn in block_names:
                    self.wedin_samples_[bn] = \
                        get_wedin_samples(X=blocks[bn],
                                          U=init_signal_svd[bn]['scores'],
                                          D=init_signal_svd[bn]['svals'],
                                          V=init_signal_svd[bn]['loadings'],
                                          rank=self.init_signal_ranks[bn],
                                          R=self.n_wedin_samples,
                                          n_jobs=self.n_jobs)

            self.wedin_sv_samples_ = len(blocks) - \
                np.array([sum(self.wedin_samples_[bn][i] **
                              2 for bn in block_names)
                          for i in range(self.n_wedin_samples)])

            # Now calculate joint matrix rank

            self.wedin_cutoff_ = np.percentile(self.wedin_sv_samples_,
                                               self.wedin_percentile)
            self.rand_cutoff_ = np.percentile(self.random_sv_samples_,
                                              self.randdir_percentile)
            self.svalsq_cutoff_ = max(self.wedin_cutoff_, self.rand_cutoff_)
            self.joint_rank_wedin_est_ = sum(joint_svals ** 2 >
                                             self.svalsq_cutoff_)
            self.joint_rank = deepcopy(self.joint_rank_wedin_est_)

        # check identifiability constraint

        if self.reconsider_joint_components:
            joint_scores, joint_svals,
            joint_loadings, self.joint_rank = \
                _reconsider_joint_components(blocks,
                                             self.sv_threshold_,
                                             joint_scores,
                                             joint_svals, joint_loadings,
                                             self.joint_rank)

        # Using rank and joint SVD, calls pca class to get joint basis
        jl = joint_loadings[:, 0:self.joint_rank]
        jsv = joint_svals[0:self.joint_rank]

        self.common = \
            pca.from_precomputed(scores=joint_scores[:,0:self.joint_rank],
                                 svals=jsv,
                                 loadings=jl,
                                 obs_names=obs_names)

        self.common.set_comp_names(['common_comp_{}'.format(i)
                                    for i in range(self.common.rank)])

        # view estimates
        block_specific = {bn: {} for bn in block_names}
        for bn in block_names:
            X = blocks[bn]  # individual matrix

            # View specific joint space creation
            # projecting X onto the joint space then compute SVD
            if self.joint_rank != 0:
                if issparse(X):  # Implement sparse JIVE later
                    raise ValueError('An input matrix is sparse. This '
                                     'functionality ' +
                                     ' is not available yet')
                else:
                    J = np.array(np.dot(joint_scores,
                                        np.dot(joint_scores.T, X)))
                    U, D, V = svd_wrapper(J, self.joint_rank)
                    if not self.store_full:
                        J = None  # kill J matrix to save memory

            else:
                U, D, V = None, None, None
                if self.store_full:
                    J = np.zeros(shape=blocks[bn].shape)
                else:
                    J = None
            # special zeros scenario
            block_specific[bn]['joint'] = {'full': J,
                                           'scores': U,
                                           'svals': D,
                                           'loadings': V,
                                           'rank': self.joint_rank}

            # Here we are creating the individual representations for
            # each view.

            # Finding the orthogonal complement to the joint matrix
            if self.joint_rank == 0:
                X_orthog = X
            else:
                X_orthog = X - np.dot(joint_scores, np.dot(joint_scores.T, X))

            # estimate individual rank using sv threshold, then compute SVD
            if self.indiv_ranks[bn] is None:
                max_rank = min(X.shape) - self.joint_rank  # saves computation
                U, D, V = svd_wrapper(X_orthog, max_rank)
                rank = sum(D > self.sv_threshold_[bn])

                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U = U[:, 0:rank]
                    D = D[0:rank]
                    V = V[:, 0:rank]

                self.indiv_ranks[bn] = rank

            # SVD on the orthogonal complement
            else:  # if user inputs rank list for individual matrices
                rank = self.indiv_ranks[bn]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = svd_wrapper(X_orthog, rank)

            # projecting X columns onto orthogonal complement of joint_scores

            if self.store_full:
                if rank == 0:
                    I_mat = np.zeros(shape=blocks[bn].shape)
                else:
                    I_mat = np.array(np.dot(U, np.dot(np.diag(D), V.T)))
            else:
                I_mat = None  # Kill I matrix to save memory

            block_specific[bn]['individual'] = {'full': I_mat,
                                                'scores': U,
                                                'svals': D,
                                                'loadings': V,
                                                'rank': rank}

            # Getting the noise matrix, E
            if self.store_full and not issparse(X):
                E = X - (J + I_mat)
            else:
                E = None
            block_specific[bn]['noise'] = E

        # save view specific estimates
        self.blocks = {}

        # Stores info for easy information checking

        for bn in block_specific.keys():
            bs_dict = block_specific[bn]
            self.blocks[bn] = \
                ViewSpecificResults(joint=bs_dict['joint'],
                                    individual=bs_dict['individual'],
                                    noise=bs_dict['noise'],
                                    block_name=bn,
                                    obs_names=obs_names,
                                    var_names=var_names[bn],
                                    m=self.centers_[bn],
                                    shape=blocks[bn].shape)

        return self

    @property
    def is_fit(self):
        if hasattr(self, "blocks"):
            return True
        else:
            return False

    @property
    def block_names(self):
        r"""
        Returns
        -------

        block_names: list
            The names of the views.

        """
        if self.is_fit:
            return list(self.blocks.keys())
        else:
            return None

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def predict(self):
        r"""
        Returns
        -------

        full: dict of dict of np.arrays
            The joint, individual, and noise full estimates for each block.

        """
        full = {}
        for bn in self.block_names:
            full[bn] = {'joint': self.blocks[bn].joint.full_,
                        'individual': self.blocks[bn].individual.full_,
                        'noise': self.blocks[bn].noise_}

        return full

    def results_dict(self):
        r"""

        Returns
        -------

        results: dict of dict of dict of np.arrays
            Returns n+1 dicts where n is the number of input views. First dict
            is a named 'common' and contains the common scores, loadings and
            rank of the views. The next n dicts represent each view. They each
            have the following keys:
                - 'joint'
                - 'individual'
                - 'noise'
                The 'joint' and 'individual' keys are dict with the following
                keys referencing their respective estimates:
                    - 'scores'
                    - 'svals'
                    - 'loadings'
                    - 'rank'
                    - 'full'

            The 'noise' key is the full noise matrix estimate of the view.

        """
        results = {}
        results['common'] = {'scores': self.common.scores_,
                             'svals': self.common.svals_,
                             'loadings': self.common.loadings_,
                             'rank': self.common.rank}

        for bn in self.block_names:
            joint = self.blocks[bn].joint
            indiv = self.blocks[bn].individual

            results[bn] = {'joint': {'scores': joint.scores_,
                                     'svals': joint.svals_,
                                     'loadings': joint.loadings_,
                                     'rank': joint.rank,
                                     'full': joint.full_},

                           'individual': {'scores': indiv.scores_,
                                          'svals': indiv.svals_,
                                          'loadings': indiv.loadings_,
                                          'rank': indiv.rank,
                                          'full': indiv.full_},

                           'noise': self.blocks[bn].noise_}

        return results

    def get_ranks(self):
        r"""
        Returns
        -------
        joint_rank: int
            The joint rank

        indiv_ranks: dict
            The individual ranks.
        """
        if not self.is_fit:
            raise ValueError("Decomposition has not yet been computed")

        joint_rank = self.common.rank
        indiv_ranks = {bn: self.blocks[bn].individual.rank for bn in
                       self.block_names}
        return joint_rank, indiv_ranks

    def data_block_heatmaps(blocks):
        r"""
        Parameters
        ----------
        blocks: dict or list of array-likes
            - blocks length: n_views
            - blocks[i] shape: (n_samples, n_features_i)
            The different views that are input. Input as data matrices.

        Returns
        -------
        fig : figure object
            Figure returned contains the heatmaps of all views
        """
        _data_block_heatmaps(blocks)

    def ajive_full_estimate_heatmaps(full_block_estimates, blocks):
        r"""
        Parameters
        ----------
        blocks: dict or list of array-likes
            - blocks length: n_views
            - blocks[i] shape: (n_samples, n_features_i)
            The different views that are input. Input as data matrices.

        full_block_estimates: dict
        Dict that is returned from the ajive.predict() function

        Returns
        -------
        fig : figure object
            Figure returned contains the full AJIVE estimates: X, J, I, E for
            all views.
        """
        _ajive_full_estimate_heatmaps(full_block_estimates, blocks)


def _dict_formatting(x):
    if hasattr(x, "keys"):
        names = list(x.keys())
        assert len(set(names)) == len(names)
    else:
        names = list(range(len(x)))
    return {n: x[n] for n in names}


def _arg_checker(blocks,
                 init_signal_ranks,
                 joint_rank,
                 indiv_ranks,
                 precomp_init_svd,
                 center):
    """
    Checks the argument inputs at different points in the code. If various
    criteria not met, errors are raised.

    """
    if hasattr(blocks, "keys"):
        blocks = _dict_formatting(blocks)
    else:
        blocks_upd = check_Xs(blocks, multiview=True)
        blocks = _dict_formatting(blocks_upd)

    block_names = list(blocks.keys())

    # check blocks have the same number of observations
    assert len(set(blocks[bn].shape[0] for bn in block_names)) == 1

    # get obs and variable names
    obs_names = list(range(list(blocks.values())[0].shape[0]))
    var_names = {}
    for bn in block_names:
        if type(blocks[bn]) == pd.DataFrame:
            obs_names = list(blocks[bn].index)
            var_names[bn] = list(blocks[bn].columns)
        else:
            var_names[bn] = list(range(blocks[bn].shape[1]))

    # format blocks
    # make sure blocks are either csr or np.array
    for bn in block_names:
        if issparse(blocks[bn]):  # TODO: allow for general linear operators
            raise ValueError('Cannot currently allow general linear operators')
        else:
            blocks[bn] = np.array(blocks[bn])

    shapes = {bn: blocks[bn].shape for bn in block_names}

    # Checking precomputed SVD
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    precomp_init_svd = _dict_formatting(precomp_init_svd)
    assert set(precomp_init_svd.keys()) == set(block_names)
    for bn in block_names:
        udv = precomp_init_svd[bn]
        if udv is not None and not hasattr(udv, "keys"):
            precomp_init_svd[bn] = {
                "scores": udv[0],
                "svals": udv[1],
                "loadings": udv[2],
            }

    # Check initial signal ranks
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    init_signal_ranks = _dict_formatting(init_signal_ranks)
    assert set(init_signal_ranks.keys()) == set(block_names)

    # signal rank must be at least one lower than the shape of the block
    for bn in block_names:
        assert 1 <= init_signal_ranks[bn]
        assert init_signal_ranks[bn] <= min(blocks[bn].shape) - 1

    # Check the joint ranks
    if joint_rank is not None and joint_rank > sum(init_signal_ranks.values()):
        raise ValueError(
            "joint_rank must be smaller than the sum of the initial signal \
            ranks"
        )

    # Check individual ranks
    if indiv_ranks is None:
        indiv_ranks = {bn: None for bn in block_names}
    indiv_ranks = _dict_formatting(indiv_ranks)
    assert set(indiv_ranks.keys()) == set(block_names)

    for k in indiv_ranks.keys():
        assert indiv_ranks[k] is None or type(indiv_ranks[k]) in [int, float]

    # Check centering
    if type(center) == bool:
        center = {bn: center for bn in block_names}
    center = _dict_formatting(center)

    return (
        blocks,
        init_signal_ranks,
        indiv_ranks,
        precomp_init_svd,
        center,
        obs_names,
        var_names,
        shapes,
    )


def _reconsider_joint_components(
    blocks, sv_threshold, joint_scores, joint_svals, joint_loadings, joint_rank
):
    """
    Checks the identifiability constraint on the joint singular values
    """

    # check identifiability constraint
    to_keep = set(range(joint_rank))
    for bn in blocks.keys():
        for j in range(joint_rank):
            # This might be joint_sv
            score = np.dot(blocks[bn].T, joint_scores[:, j])
            sv = np.linalg.norm(score)

            # if sv is below the threshold for any data block remove j
            if sv < sv_threshold[bn]:
                print("removing column " + str(j))
                to_keep.remove(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    joint_scores = joint_scores[:, list(to_keep)]
    joint_loadings = joint_loadings[:, list(to_keep)]
    joint_svals = joint_svals[list(to_keep)]
    return joint_scores, joint_svals, joint_loadings, joint_rank


class ViewSpecificResults(object):
    """
    Contains the view specific results.

    Parameters
    ----------
    joint: dict
        The view specific joint PCA.

    individual: dict
        The view specific individual PCA.

    noise: array-like
        The noise matrix estimate.

    obs_names: array-like, default = None
        Observation names.

    var_names: array-like, default = None
        Variable names for this view.

    block_name: str, default = None
        Name of this view.

    m: array-like, default = None
        The vector used to column mean center this view.


    Attributes
    ----------
    joint: mvlearn.ajive.pca.pca
        View specific joint PCA.
        Has an extra attribute joint.full_ which contains the full view
        joint estimate.

    individual: mvlearn.ajive.pca.pca
        View specific individual PCA.
        Has an extra attribute individual.full_ which contains the full view
        joint estimate.

    noise: array-like
        The full noise view estimate.

    block_name:
        Name of this view.

    """

    def __init__(
        self,
        joint,
        individual,
        noise,
        obs_names=None,
        var_names=None,
        block_name=None,
        m=None,
        shape=None,
    ):

        self.joint = pca.from_precomputed(
            n_components=joint["rank"],
            scores=joint["scores"],
            loadings=joint["loadings"],
            svals=joint["svals"],
            obs_names=obs_names,
            var_names=var_names,
            m=m,
            shape=shape,
        )

        if joint["rank"] != 0:
            self.joint.set_comp_names(
                ["joint_comp_{}".format(i) for i in range(self.joint.rank)]
            )

        if joint["full"] is not None:
            self.joint.full_ = pd.DataFrame(
                joint["full"], index=obs_names, columns=var_names
            )
        else:
            self.joint.full_ = None

        self.individual = pca.from_precomputed(
            n_components=individual["rank"],
            scores=individual["scores"],
            loadings=individual["loadings"],
            svals=individual["svals"],
            obs_names=obs_names,
            var_names=var_names,
            m=m,
            shape=shape,
        )
        if individual["rank"] != 0:
            self.individual.set_comp_names(
                [
                    "indiv_comp_{}".format(i)
                    for i in range(self.individual.rank)
                ]
            )

        if individual["full"] is not None:
            self.individual.full_ = pd.DataFrame(
                individual["full"], index=obs_names, columns=var_names
            )
        else:
            self.individual.full_ = None

        if noise is not None:
            self.noise_ = pd.DataFrame(
                noise, index=obs_names, columns=var_names
            )
        else:
            self.noise_ = None

        self.block_name = block_name

    def __repr__(self):
        return "Block: {}, individual rank: {}, joint rank: {}".format(
            self.block_name, self.individual.rank, self.joint.rank
        )
