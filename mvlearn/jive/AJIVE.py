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
import warnings
from scipy.sparse import issparse
from copy import deepcopy
from sklearn.externals.joblib import load, dump
import pandas as pd
from mvlearn.embed.base import BaseEmbed
from mvlearn.utils.utils import check_Xs
import warnings

from .utils import svd_wrapper, centering
from .wedin_bound import get_wedin_samples
from .random_direction import sample_randdir
from .diagnostic_plot import plot_joint_diagnostic
from .PCA import PCA


class AJIVE(object):
    """
    An implementation of Angle-based Joint and Individual Variation Explained.
    This algorithm takes multiple input views with the same number of samples
    and decomposes them into 3 distinct matrices representing:
        - Individual variation of each particular view
        - Joint variation shared by all views
        - Noise

    Parameters
    ----------
    init_signal_ranks: {list, dict}
        The initial signal ranks.

    joint_rank: {None, int}
        Rank of the joint variation matrix. If None, will estimate the 
        joint rank. Otherwise, will use provided joint rank.

    indiv_ranks: {list, dict, None}
        Ranks of individual variation matrices. If None, will estimate the 
        individual ranks. Otherwise, will use provided individual ranks.

    center: {bool, None}
        Centers matrix. If None, will not center.

    reconsider_joint_components: bool
        Triggers reconsider_joint_components function

    wedin_percentile: int, default=5
        Percentile used for wedin (lower) bound cutoff for squared 
        singular values used to estimate joint rank.

    n_wedin_samples: int, default=1000
        Number of wedin bound samples to draw.

    precomp_wedin_samples {None, dict of array-like, list of array-like}
        Wedin samples that are precomputed for each view.

    randdir_percentile: int, default=95
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank.

    n_randdir_samples: int, default=1000
        Number of random direction samples to draw.

    precomp_randdir_samples {None,  array-like}
        Precomputed random direction samples.

    n_jobs: int, None
        Number of jobs for parallel processing wedin samples and random
        direction samples using sklearn.externals.joblib.Parallel.
        If None, will not use parallel processing.

    Attributes
    ----------

    common: mvlearn.jive.PCA.PCA
        The common joint space found using PCA class in same directory

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
            r = "AJIVE, joint rank: {}".format(self.common.rank)
            for bn in self.block_names:
                indiv_rank = self.blocks[bn].individual.rank
                r += ", block {} indiv rank: {}".format(bn, indiv_rank)
            return r

        else:
            return "No data has been fitted yet"

    def fit(self, blocks, precomp_init_svd=None):
        """
        Fits the AJIVE decomposition.

        Parameters
        ----------
        blocks: {list of array-likes or numpy.ndarray, dict}
            - blocks length: n_views
            - blocks[i] shape: (n_samples, n_features_i) 
            The different views that are input. Input as data matrices. 
            If dict, will name blocks by keys, otherwise blocks are named by 
            0, 1, ...K. 
            
        precomp_init_svd: {list, dict, None}, optional
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
        num_obs = list(blocks.values())[0].shape[0] #number of views

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
                U, D, V = svd_wrapper(blocks[bn], \
                                      self.init_signal_ranks[bn] + 1)
            # If precomputed return values already found
            else:
                U = precomp_init_svd[bn]["scores"]
                D = precomp_init_svd[bn]["svals"]
                V = precomp_init_svd[bn]["loadings"]

            # The SV threshold is halfway between the init_signal_ranks[bn]th
            # and init_signal_ranks[bn] + 1 st singular value.
            self.sv_threshold_[bn] = (D[self.init_signal_ranks[bn] - 1] \
                                      + D[self.init_signal_ranks[bn]])/2

            init_signal_svd[bn] = {'scores': \
                           U[:, 0:self.init_signal_ranks[bn]],
                                   'svals': \
                                   D[0:self.init_signal_ranks[bn]],
                                   'loadings': \
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
                self.random_sv_samples_ = \
                    sample_randdir(num_obs,
                                   signal_ranks=\
                                   list(self.init_signal_ranks.values()),
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
                np.array([sum(self.wedin_samples_[bn][i] ** \
                              2 for bn in block_names)
                          for i in range(self.n_wedin_samples)])

            # Now calculate joint matrix rank

            self.wedin_cutoff_ = np.percentile(self.wedin_sv_samples_,
                                               self.wedin_percentile)
            self.rand_cutoff_ = np.percentile(self.random_sv_samples_,
                                              self.randdir_percentile)
            self.svalsq_cutoff_ = max(self.wedin_cutoff_, self.rand_cutoff_)
            self.joint_rank_wedin_est_ = sum(joint_svals ** 2 > \
                                             self.svalsq_cutoff_)
            self.joint_rank = deepcopy(self.joint_rank_wedin_est_)

        # check identifiability constraint

        if self.reconsider_joint_components:
            joint_scores, joint_svals, joint_loadings, self.joint_rank = \
                reconsider_joint_components(blocks, self.sv_threshold_,
                                            joint_scores, 
                                            joint_svals, joint_loadings,
                                            self.joint_rank)

        # Using rank and joint SVD, calls PCA class to get joint basis
        self.common = \
        PCA.from_precomputed(scores=joint_scores[:, 0:self.joint_rank],
                                           svals=\
                                           joint_svals[0:self.joint_rank], 
                                           loadings=joint_loadings\
                                           [:, 0:self.joint_rank],
                                           obs_names=obs_names)

        self.common.set_comp_names(['common_comp_{}'.format(i)
                                    for i in range(self.common.rank)])


        # view estimates
        
        block_specific = {bn: {} for bn in block_names}
        for bn in block_names:
            X = blocks[bn] #individual matrix
            
            # View specific joint space creation
            # projecting X onto the joint space then compute SVD
            if self.joint_rank != 0:
                if issparse(X):  # Implement sparse JIVE later
                    raise ValueError('An input matrix is sparse. This ' 
                                     'functionality '+
                                     ' is not available yet')                    
                else:
                    J = np.array(np.dot(joint_scores, \
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
            #special zeros scenario
            block_specific[bn]['joint'] = {'full': J,
                                           'scores': U,
                                           'svals': D,
                                           'loadings': V,
                                           'rank': self.joint_rank}


            # project X onto the orthogonal complement of the joint space,
            # estimate the individual rank, then compute SVD
            # project X columns onto orthogonal complement of joint_scores

            # Here we are creating the individual representations for 
            # each view. 
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

            else:  # indiv_rank has been provided by the user
                rank = self.indiv_ranks[bn]
                if rank == 0:
                    U, D, V = None, None, None
                else:
                    U, D, V = svd_wrapper(X_orthog, rank)

            if self.store_full:
                if rank == 0:
                    I = np.zeros(shape=blocks[bn].shape)
                else:
                    I = np.array(np.dot(U, np.dot(np.diag(D), V.T)))
            else:
                I = None  # Kill I matrix to save memory

            block_specific[bn]['individual'] = {'full': I,
                                            'scores': U,
                                            'svals': D,
                                            'loadings': V,
                                            'rank': rank}

            ###################################
            # step 3.3: estimate noise matrix #
            ###################################

            if self.store_full and not issparse(X):
                E = X - (J + I)
            else:
                E = None
            block_specific[bn]['noise'] = E

        # save block specific estimates
        self.blocks = {}

        for bn in block_specific.keys():
            self.blocks[bn] = BlockSpecificResults(joint=block_specific[bn]['joint'],
                                                   individual=block_specific[bn]['individual'],
                                                   noise=block_specific[bn]['noise'],
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
        if self.is_fit:
            return list(self.blocks.keys())
        else:
            return None

    def plot_joint_diagnostic(self, fontsize=20):
        """
        Plots joint rank threshold diagnostic plot
        """

        plot_joint_diagnostic(joint_svals=self.all_joint_svals_,
                              wedin_sv_samples=self.wedin_sv_samples_,
                              min_signal_rank=min(self.init_signal_ranks.values()),
                              random_sv_samples=self.random_sv_samples_,
                              wedin_percentile=self.wedin_percentile,
                              random_percentile=self.randdir_percentile,
                              fontsize=fontsize)

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def get_full_block_estimates(self):
        """

        Output
        ------
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
        """
        Returns all estimates as a dicts.

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
        """
        Output
        ------
        joint_rank (int): the joint rank

        indiv_ranks (dict): the individual ranks.
        """
        if not self.is_fit:
            raise ValueError("Decomposition has not yet been computed")

        joint_rank = self.common.rank
        indiv_ranks = {bn: self.blocks[bn].individual.rank for bn in self.block_names}
        return joint_rank, indiv_ranks


def _dict_formatting(x):
    if hasattr(x, "keys"):
        names = list(x.keys())
        assert len(set(names)) == len(names)
    else:
        names = list(range(len(x)))
    return {n: x[n] for n in names}


def _arg_checker(blocks, init_signal_ranks, joint_rank, indiv_ranks,
                precomp_init_svd, center):
    """
    Checks the argument inputs at different points in the code. If various 
    criteria not met, errors are raised.
        
    """
    # TODO: document
    # TODO: change assert to raise ValueError with informative message

    ##########
    # blocks #
    ##########

    blocks = _dict_formatting(blocks)
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

    ####################
    # precomp_init_svd #
    ####################
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

    # TODO: check either None or SVD provided
    # TODO: check correct SVD formatting
    # TODO: check SVD ranks are the same
    # TODO: check SVD rank is at least init_signal_ranks + 1

    #####################
    # init_signal_ranks #
    #####################
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    init_signal_ranks = _dict_formatting(init_signal_ranks)
    print(set(init_signal_ranks.keys()))
    print(block_names)
    assert set(init_signal_ranks.keys()) == set(block_names)

    # initial signal rank must be at least one lower than the shape of the block
    for bn in block_names:
        assert 1 <= init_signal_ranks[bn]
        assert init_signal_ranks[bn] <= min(blocks[bn].shape) - 1

    ##############
    # joint_rank #
    ##############
    if joint_rank is not None and joint_rank > sum(init_signal_ranks.values()):
        raise ValueError(
            "joint_rank must be smaller than the sum of the initial signal ranks"
        )

    ###############
    # indiv_ranks #
    ###############
    if indiv_ranks is None:
        indiv_ranks = {bn: None for bn in block_names}
    indiv_ranks = _dict_formatting(indiv_ranks)
    assert set(indiv_ranks.keys()) == set(block_names)

    for k in indiv_ranks.keys():
        assert indiv_ranks[k] is None or type(indiv_ranks[k]) in [int, float]
        # TODO: better check for numeric

    ##########
    # center #
    ##########
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


def reconsider_joint_components(
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
                # TODO: should probably keep track of this
                print("removing column " + str(j))
                to_keep.remove(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    joint_scores = joint_scores[:, list(to_keep)]
    joint_loadings = joint_loadings[:, list(to_keep)]
    joint_svals = joint_svals[list(to_keep)]
    return joint_scores, joint_svals, joint_loadings, joint_rank


class BlockSpecificResults(object):
    """
    Contains the block specific results.

    Parameters
    ----------
    joint: dict
        The block specific joint PCA.

    individual: dict
        The block specific individual PCA.

    noise: array-like
        The noise matrix estimate.

    obs_names: None, array-like
        Observation names.

    var_names: None, array-like
        Variable names for this block.

    block_name: None, int, str
        Name of this block.

    m: None, array-like
        The vector used to column mean center this block.


    Attributes
    ----------
    joint: jive.PCA.PCA
        Block specific joint PCA.
        Has an extra attribute joint.full_ which contains the full block
        joint estimate.

    individual: jive.PCA.PCA
        Block specific individual PCA.
        Has an extra attribute individual.full_ which contains the full block
        joint estimate.


    noise: array-like
        The full noise block estimate.

    block_name:
        Name of this block.

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

        self.joint = PCA.from_precomputed(
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

        self.individual = PCA.from_precomputed(
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

