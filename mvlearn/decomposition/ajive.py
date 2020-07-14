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
import pandas as pd
from .base import BaseDecomposer
from ..utils.utils import check_Xs
from ..embed.utils import select_dimension
from .ajive_utils.block_visualization import _data_block_heatmaps, \
    _ajive_full_estimate_heatmaps
from .ajive_utils.utils import svd_wrapper, centering
from .ajive_utils.wedin_bound import get_wedin_samples
from .ajive_utils.random_direction import sample_randdir
from .ajive_utils.pca import pca, ViewSpecificResults


class AJIVE(BaseDecomposer):
    r"""
    An implementation of Angle-based Joint and Individual Variation Explained
    [#1ajive]_. This algorithm takes multiple views and decomposes them into 3
    distinct matrices representing:
        - Low rank approximation of individual variation within each view
        - Low rank approximation of joint variation between views
        - Residual noise
    An important note, AJIVE can handle any number of views, not just two.

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

    sv_threshold_ : dict
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

        self.center_ = center

        self.wedin_percentile = wedin_percentile
        self.n_wedin_samples = n_wedin_samples
        self.wedin_samples_ = precomp_wedin_samples
        if precomp_wedin_samples is not None:
            self.n_wedin_samples = len(precomp_wedin_samples[0])

        self.randdir_percentile = randdir_percentile
        self.n_randdir_samples = n_randdir_samples
        self.random_sv_samples_ = precomp_randdir_samples
        if precomp_randdir_samples is not None:
            self.n_randdir_samples = len(precomp_randdir_samples)

        self.reconsider_joint_components = reconsider_joint_components

        self.store_full = store_full

    def __repr__(self):

        if self.is_fit_:
            r = "joint rank: {}".format(self.common_.rank)
            for bn in self.block_names:
                indiv_rank = self.blocks_[bn].individual.rank
                r += ", block {} indiv rank: {}".format(bn, indiv_rank)
            return r

        else:
            return "No data has been fitted yet"

    def fit(self, Xs, view_names=None, precomp_init_svd=None):
        r"""
        Learns the AJIVE decomposition from Xs.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The different views that are input. Input as data matrices.

        precomp_init_svd: dict or list
            Precomputed initial SVD. Must have one entry for each data block.
            The SVD should be an ordered list of 3 matrices (scores, svals,
            loadings), see output of .ajive_utils/utils/svd_wrapper for
            formatting details.

        view_names: array-like, default = None
            Optional. The names of the views. If no input, the views will be
            names 1,2,...n_views.

        Returns
        -------
        self : returns an instance of self.
        """
        Xs = check_Xs(Xs, multiview=True)
        self.init_signal_ranks_ = self.init_signal_ranks
        if self.init_signal_ranks_ is None:
            self.init_signal_ranks_ = []
            for X in Xs:
                elbows, _ = select_dimension(X, n_elbows=self.n_elbows)
                self.init_signal_ranks_.append(elbows[-1])

        Xs, self.init_signal_ranks_, self.indiv_ranks_, precomp_init_svd,\
            self.center_, obs_names, var_names, self.shapes_ =\
            _arg_checker(Xs, view_names,
                         self.init_signal_ranks_,
                         self.joint_rank,
                         self.indiv_ranks,
                         precomp_init_svd,
                         self.center_,
                         )

        block_names = list(Xs.keys())
        num_obs = list(Xs.values())[0].shape[0]  # number of views

        # centering views
        self.centers_ = {}
        for bn in block_names:
            Xs[bn], self.centers_[bn] = centering(Xs[bn],
                                                  method=self.center_[bn])

        # SVD to extract signal on each view

        init_signal_svd = {}
        self.sv_threshold_ = {}
        for bn in block_names:

            # compute SVD with rank init_signal_ranks[bn] + 1 for view
            if precomp_init_svd[bn] is None:
                # signal rank + 1 to get individual rank sv threshold
                U, D, V = svd_wrapper(Xs[bn],
                                      self.init_signal_ranks_[bn] + 1)
            # If precomputed return values already found
            else:
                U = precomp_init_svd[bn]["scores"]
                D = precomp_init_svd[bn]["svals"]
                V = precomp_init_svd[bn]["loadings"]

            # The SV threshold is halfway between the init_signal_ranks[bn]th
            # and init_signal_ranks[bn] + 1 st singular value.
            self.sv_threshold_[bn] = (D[self.init_signal_ranks_[bn] - 1]
                                      + D[self.init_signal_ranks_[bn]])/2

            init_signal_svd[bn] = {'scores':
                                   U[:, 0:self.init_signal_ranks_[bn]],
                                   'svals':
                                   D[0:self.init_signal_ranks_[bn]],
                                   'loadings':
                                   V[:, 0:self.init_signal_ranks_[bn]]}

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
                init_rank_list = list(self.init_signal_ranks_.values())
                self.random_sv_samples_ = \
                    sample_randdir(num_obs,
                                   signal_ranks=init_rank_list,
                                   R=self.n_randdir_samples)

            # if the wedin samples are not already provided compute them
            if self.wedin_samples_ is None:
                self.wedin_samples_ = {}
                for bn in block_names:
                    self.wedin_samples_[bn] = \
                        get_wedin_samples(X=Xs[bn],
                                          U=init_signal_svd[bn]['scores'],
                                          D=init_signal_svd[bn]['svals'],
                                          V=init_signal_svd[bn]['loadings'],
                                          rank=self.init_signal_ranks_[bn],
                                          R=self.n_wedin_samples)
            else:
                self.wedin_samples_ = {
                    bn: wc for bn, wc in zip(self.wedin_samples_, block_names)
                    }

            self.wedin_sv_samples_ = len(Xs) - \
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
            self.joint_rank_ = deepcopy(self.joint_rank_wedin_est_)
        else:
            self.joint_rank_ = deepcopy(self.joint_rank)

        # check identifiability constraint

        if self.reconsider_joint_components:
            joint_scores, joint_svals, joint_loadings, self.joint_rank_ = \
                _reconsider_joint_components(Xs,
                                             self.sv_threshold_,
                                             joint_scores,
                                             joint_svals, joint_loadings,
                                             self.joint_rank_)

        # Using rank and joint SVD, calls pca class to get joint basis
        jl = joint_loadings[:, 0:self.joint_rank_]
        jsv = joint_svals[0:self.joint_rank_]

        self.common_ = \
            pca.from_precomputed(scores=joint_scores[:, 0:self.joint_rank_],
                                 svals=jsv,
                                 loadings=jl,
                                 obs_names=obs_names)

        self.common_.set_comp_names(['common_comp_{}'.format(i)
                                    for i in range(self.common_.rank)])

        # view estimates
        block_specific = {bn: {} for bn in block_names}
        for bn in block_names:
            X = Xs[bn]  # individual matrix

            # View specific joint space creation
            # projecting X onto the joint space then compute SVD
            if self.joint_rank_ != 0:
                if issparse(X):  # Implement sparse JIVE later
                    raise ValueError('An input matrix is sparse. This '
                                     'functionality ' +
                                     ' is not available yet')
                else:
                    J = np.array(np.dot(joint_scores,
                                        np.dot(joint_scores.T, X)))
                    U, D, V = svd_wrapper(J, self.joint_rank_)
                    if not self.store_full:
                        J = None  # kill J matrix to save memory

            else:
                U, D, V = None, None, None
                if self.store_full:
                    J = np.zeros(shape=Xs[bn].shape)
                else:
                    J = None
            # special zeros scenario
            block_specific[bn]['joint'] = {'full': J,
                                           'scores': U,
                                           'svals': D,
                                           'loadings': V,
                                           'rank': self.joint_rank_}

            # Here we are creating the individual representations for
            # each view.

            # Finding the orthogonal complement to the joint matrix
            if self.joint_rank_ == 0:
                X_orthog = X
            else:
                X_orthog = X - np.dot(joint_scores, np.dot(joint_scores.T, X))

            # estimate individual rank using sv threshold, then compute SVD
            if self.indiv_ranks_[bn] is None:
                max_rank = min(X.shape) - self.joint_rank_  # saves computation
                U, D, V = svd_wrapper(X_orthog, max_rank)
                rank = sum(D > self.sv_threshold_[bn])

                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U = U[:, 0:rank]
                    D = D[0:rank]
                    V = V[:, 0:rank]

                self.indiv_ranks_[bn] = rank

            # SVD on the orthogonal complement
            else:  # if user inputs rank list for individual matrices
                rank = self.indiv_ranks_[bn]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = svd_wrapper(X_orthog, rank)

            # projecting X columns onto orthogonal complement of joint_scores

            if self.store_full:
                if rank == 0:
                    I_mat = np.zeros(shape=Xs[bn].shape)
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
        self.blocks_ = {}

        # Stores info for easy information checking

        for bn in block_specific.keys():
            bs_dict = block_specific[bn]
            self.blocks_[bn] = \
                ViewSpecificResults(joint=bs_dict['joint'],
                                    individual=bs_dict['individual'],
                                    noise=bs_dict['noise'],
                                    block_name=bn,
                                    obs_names=obs_names,
                                    var_names=var_names[bn],
                                    m=self.centers_[bn],
                                    shape=Xs[bn].shape)

        return self

    @property
    def is_fit_(self):
        if hasattr(self, "blocks_"):
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
        if self.is_fit_:
            return list(self.blocks_.keys())
        else:
            return None

    def transform(self, Xs=None, return_dict=False):
        r"""

        Returns the joint, individual, and noise components of each view from
        the fitted decomposition. Only works on the data inputted in `fit`.

        Parameters
        ----------

        Xs : ignored
            Not used but included for API consistency. Predictions come from
            the fitted data.

        return_dict: bool, default = False
            If True, return is in dictionary format, if False, return is in
            list format.

        Returns
        -------

        full: list of lists of np.arrays or dict of dicts of np.arrays holding
            the joint, individual, and noise full estimates for each block.
            In list format the inital indices represent each view. Within these
            views, the first index represents the view's full joint estimate,
            the second index represents the view's full individual estimate,
            and the third index represents the view's noise matrix. The
            dictionary format returns the same matrices as dictionaries with
            the top-level dictionary keys representing views and each view
            dictionary keys representing the type of matrix ('joint',
            'individual', 'noise')

        """
        full_dict = {}
        full_list = []

        if return_dict is True:
            for bn in self.block_names:
                indivi_tot_dict = self.blocks_[bn].individual.full_
                full_dict[bn] = {'joint': self.blocks_[bn].joint.full_,
                                 'individual': indivi_tot_dict,
                                 'noise': self.blocks_[bn].noise_}
            return full_dict

        else:
            for bn in self.block_names:
                full_list.append([self.blocks_[bn].joint.full_,
                                  self.blocks_[bn].individual.full_,
                                  self.blocks_[bn].noise_])
            return full_list

    def results_dict(self):
        r"""

        Returns a summary of the fitted results in a dictionary.

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
        results['common'] = {'scores': self.common_.scores_,
                             'svals': self.common_.svals_,
                             'loadings': self.common_.loadings_,
                             'rank': self.common_.rank}

        for bn in self.block_names:
            joint = self.blocks_[bn].joint
            indiv = self.blocks_[bn].individual

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

                           'noise': self.blocks_[bn].noise_}

        return results

    def get_ranks(self):
        r"""
        Returns the joint and individual ranks of the decomposition.

        Returns
        -------
        joint_rank: int
            The joint rank

        indiv_ranks: dict
            The individual ranks.
        """
        if not self.is_fit_:
            raise ValueError("Decomposition has not yet been computed")

        joint_rank = self.common_.rank
        indiv_ranks = {bn: self.blocks_[bn].individual.rank for bn in
                       self.block_names}
        return joint_rank, indiv_ranks


def data_block_heatmaps(Xs):
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


def ajive_full_estimate_heatmaps(Xs, full_block_estimates, names=None):
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


def _dict_formatting_first(x, view_names):
    if view_names is not None:
        assert len(set(view_names)) == len(view_names)
        return {view_names[i]: x[i] for i in np.arange(len(view_names))}
    else:
        view_names = list(range(len(x)))
        return {n: x[n] for n in view_names}


def _dict_formatting(x, names):
    if hasattr(x, 'keys'):
        vals = list(x.values())
        assert len(set(names)) == len(names)
        return {names[i]: vals[i] for i in np.arange(len(names))}
    else:
        return {names[i]: x[i] for i in np.arange(len(names))}


def _names_checker(x, names):
    if names is None:
        return names

    if isinstance(names, (list, pd.core.series.Series, np.ndarray)):
        if len(x) == len(names):
            return list(names)
        else:
            raise ValueError(
                    "The number of view inputs must match the number of name \
                    inputs"
                             )
    else:
        raise ValueError('view_names must be an array-like input')


def _arg_checker(Xs,
                 view_names,
                 init_signal_ranks,
                 joint_rank,
                 indiv_ranks,
                 precomp_init_svd,
                 center
                 ):
    """
    Checks the argument inputs at different points in the code. If various
    criteria not met, errors are raised.

    """
    names = _names_checker(Xs, view_names)
    Xs = _dict_formatting_first(Xs, names)

    block_names = list(Xs.keys())

    # check blocks have the same number of observations
    assert len(set(Xs[bn].shape[0] for bn in block_names)) == 1

    # get obs and variable names
    obs_names = list(range(list(Xs.values())[0].shape[0]))
    var_names = {}
    for bn in block_names:
        if type(Xs[bn]) == pd.DataFrame:
            obs_names = list(Xs[bn].index)
            var_names[bn] = list(Xs[bn].columns)
        else:
            var_names[bn] = list(range(Xs[bn].shape[1]))

    # format blocks
    # make sure blocks are either csr or np.array
    for bn in block_names:
        if issparse(Xs[bn]):  # TODO: allow for general linear operators
            raise ValueError('Cannot currently allow general linear operators')
        else:
            Xs[bn] = np.array(Xs[bn])

    shapes = {bn: Xs[bn].shape for bn in block_names}

    # Checking precomputed SVD
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    precomp_init_svd = _dict_formatting(precomp_init_svd, block_names)
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
    init_signal_ranks = _dict_formatting(init_signal_ranks, block_names)
    assert set(init_signal_ranks.keys()) == set(block_names)

    # signal rank must be at least one lower than the shape of the block
    for bn in block_names:
        assert 1 <= init_signal_ranks[bn]
        assert init_signal_ranks[bn] <= min(Xs[bn].shape) - 1

    # Check the joint ranks
    if joint_rank is not None and joint_rank > sum(init_signal_ranks.values()):
        raise ValueError(
            "joint_rank must be smaller than the sum of the initial signal \
            ranks"
        )

    # Check individual ranks
    if indiv_ranks is None:
        indiv_ranks = {bn: None for bn in block_names}
    indiv_ranks = _dict_formatting(indiv_ranks, block_names)
    assert set(indiv_ranks.keys()) == set(block_names)

    for k in indiv_ranks.keys():
        assert indiv_ranks[k] is None or type(indiv_ranks[k]) in [int, float]

    # Check centering
    if type(center) == bool:
        center = {bn: center for bn in block_names}
    center = _dict_formatting(center, block_names)

    return (
        Xs,
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
