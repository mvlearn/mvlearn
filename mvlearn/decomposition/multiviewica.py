# BSD 3-Clause License
# Copyright (c) 2020, Hugo RICHARD and Pierre ABLIN
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Modified from source package https://github.com/hugorichard/multiviewica

import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils.extmath import randomized_svd
from joblib import Parallel, delayed
from picard import picard
from multiviewica import multiviewica
from multiviewica import permica
from ..preprocessing.repeat import ViewTransformer
from sklearn.decomposition import PCA

from .baseica import BaseICA


class MultiviewICA(BaseICA):
    r"""
    Multiview ICA for which views share a common source but separate mixing
    matrices.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset.

    noise : float, default=1.0
        Gaussian noise level

    max_iter : int, default=1000
        Maximum number of iterations to perform

    init : {'permica', 'groupica'} or np array of shape
        (n_groups, n_components, n_components), default='permica'
        If permica: initialize with perm ICA, if groupica, initialize with
        group ica. Else, use the provided array to initialize.

    preproc: 'pca' or a ViewTransformer-like instance,
        default='pca'
        Preprocessing method to use to reduce data.
        If "pca", performs PCA separately on each view to reduce dimension
        of each view.
        Otherwise the dimension reduction is performed using the transform
        method of the ViewTransformer-like object. This instance also needs
        an inverse transform method to recover original data from reduced data.

    multiview_output : bool, optional (default True)
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    random_state : int, RandomState instance or None, default=None
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.

    tol : float, default=1e-3
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.

    verbose : bool, default=False
        Print information

    n_jobs : int (positive), default=None
        The number of jobs to run in parallel. `None` means 1 job, `-1`
        means using all processors.

    Attributes
    ----------
    preproc : ViewTransformer-like instance
        The fitted instance used for preprocessing
    W_list : np array of shape (n_groups, n_components, n_components)
        The unmixing matrices to apply on preprocessed data

    See also
    --------
    groupica
    permica

    Notes
    -----
    Given each view :math:`X_i` It optimizes:

        .. math::
            l(W) = \frac{1}{T} \sum_{t=1}^T [\sum_k log(cosh(Y_{avg,k,t}))
            + \sum_i l_i(X_{i,.,t})]

    where

        .. math::
            l _i(X_{i,.,t}) = - log(|W_i|) + 1/(2 \sigma) ||X_{i,.,t}W_i -
            Y_{avg,.,t}||^2,

    :math:`W_i` is the mixing matrix for view  :math:`i`,
    :math:`Y_{avg} = \frac{1}{n} \sum_{i=1}^n X_i W_i`, and :math:`\sigma`
    is the noise level.

    References
    ----------
    .. [#1mvica] Hugo Richard, Luigi Gresele, Aapo Hyvärinen, Bertrand Thirion,
        Alexandre Gramfort, Pierre Ablin. Modeling Shared Responses in
        Neuroimaging Studies through MultiView ICA. arXiv 2020.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import MultiviewICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = MultiviewICA(n_components=3, max_iter=10)
    >>> sources = ica.fit_transform(Xs)
    >>> print(sources.shape)
    (6, 2000, 3)
    """

    def __init__(
        self,
        n_components=None,
        noise=1.0,
        max_iter=1000,
        init="permica",
        dimension_reduction="pca",
        multiview_output=True,
        random_state=None,
        tol=1e-3,
        verbose=False,
        n_jobs=30,
    ):
        self.n_components = n_components
        self.noise = noise
        self.max_iter = max_iter
        self.multiview_output = multiview_output
        self.init = init
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit_(self, Xs, y=None):
        r"""
        Fits the model to the views Xs.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Training data to recover a source and unmixing matrices from.
        y : ignored

        Returns
        -------
        self : returns an instance of itself.
        """
        return multiviewica(
            Xs,
            noise=self.noise,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
            tol=self.tol,
            verbose=self.verbose,
        )
