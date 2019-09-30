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

from tqdm import tqdm


class GCCA(BaseEmbed):
    """
    An implementation of Generalized Canonical Correalation Analysis.Computes individual 
    projections into a common subspace such that the correlations between pairwise projections 
    are minimized (ie. maximize pairwise correlation). Reduces to CCA in the case of two samples.
    
    See https://www.sciencedirect.com/science/article/pii/S1053811912001644?via%3Dihub
    for relevant details.
    """

    def __init__(self):
        super().__init__()
        self._projection_mats = None

    def _preprocess(self, X):
        """
        Subtracts the row means and divides by the row standard deviations.
        Then subtracts column means.
        
        Parameters
        ----------
        X : array-like, shape (n_observations, n_features)
            The data to preprocess

        Returns
        -------
        X2 : preprocessed data matrix
        """

        # Mean along rows using sample mean and sample std
        X2 = stats.zscore(X, axis=1, ddof=1)
        # Mean along columns
        mu = np.mean(X2, axis=0)
        X2 -= mu
        return X2

    def fit(
        self,
        Xs,
        percent_var=0.9,
        rank_tolerance=None,
        n_components=None,
        tall=False,
        verbose=False,
    ):
        """
        
        Parameters
        ----------
        Xs : array-like, shape (n_views, n_samples, n_features)
            The data to fit to. Each sample will receive its own embedding.
        percent_var : percent, default=0.9
            Explained variance for rank selection during initial SVD of each sample.
        rank_tolerance : float, optional, default=None
            Singular value threshold for rank selection during initial SVD of each sample.
        n_components : int (postivie), optional, default=None
            Rank to truncate to during initial SVD of each sample.
        tall : boolean, default=False
            Set to true if n_samples > n_features, speeds up SVD
        """
        Xs = check_Xs(Xs)
        n = Xs.shape[1]

        data = [self._preprocess(x) for x in Xs]

        Uall = []
        Sall = []
        Vall = []
        ranks = []

        for x in tqdm(data, disable=(not verbose)):
            # Preprocess
            x[np.isnan(x)] = 0

            # compute the SVD of the data
            if tall:
                v, s, ut = linalg.svd(x.T, full_matrices=False)
            else:
                u, s, vt = linalg.svd(x, full_matrices=False)
                ut = u.T
                v = vt.T

            Sall.append(s)
            Vall.append(v)
            # Dimensions to reduce to
            if rank_tolerance:
                rank = sum(s > rank_tolerance)
            elif n_components:
                rank = n_components
            else:
                s2 = np.square(s)
                rank = sum(np.cumsum(s2 / sum(s2)) < percent_var) + 1
            ranks.append(rank)

            u = ut.T[:, :rank]
            Uall.append(u)

        d = min(ranks)

        # Create a concatenated view of Us
        Uall_c = np.concatenate(Uall, axis=1)

        _, _, VV = svds(Uall_c, d)
        VV = np.flip(VV.T, axis=1)
        VV = VV[:, : min([d, VV.shape[1]])]

        # SVDS the concatenated Us
        idx_end = 0
        projXs = []
        projection_mats = []
        for i in range(len(data)):
            idx_start = idx_end
            idx_end = idx_start + ranks[i]
            VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)

            # Compute the canonical projections
            A = np.sqrt(n - 1) * Vall[i][:, : ranks[i]]
            A = A @ (linalg.solve(np.diag(Sall[i][: ranks[i]]), VVi))
            projXs.append(data[i] @ A)
            projection_mats.append(A)

        self._projection_mats = projection_mats
        self._ranks = ranks

        return self

    def transform(self, Xs, view_idx=None):
        """
        Embeds data matrix(s) using the fitted projection matrices

        Parameters
        ----------
        Xs : array-like, shape (n_views, n_samples, n_features)
            The data to embed based on the prior fit function
        view_idx: int
            The index of the view whose projection to use on Xs. For a single view.

        Returns
        -------
        X_transformed : array-like
            2D if view_idx not None, otherwise same shape as Xs
        """
        Xs = check_Xs(Xs)
        if view_idx is not None:
            try:
                return self._preprocess(Xs[0]) @ self._projection_mats[view_idx]
            except IndexError:
                print(f"view_idx: {view_idx} invalid")
        else:
            return np.array(
                [
                    self._preprocess(x) @ proj
                    for x, proj in zip(Xs, self._projection_mats)
                ]
            )
