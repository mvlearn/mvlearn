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


class GCCA(BaseEmbed):
    """
    An implementation of Generalized Canonical Correalation Analysis. Computes
    individual projections into a common subspace such that the correlations
    between pairwise projections are minimized (ie. maximize pairwise
    correlation). Reduces to CCA in the case of two samples.

    See https://www.sciencedirect.com/science/article/pii/S1053811912001644?
    via%3Dihub
    for relevant details.
    """

    def __init__(self):
        super().__init__()
        self._projection_mats = None

    def _center(self, X):
        """
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

    def fit(
        self,
        Xs,
        fraction_var=0.9,
        sv_tolerance=None,
        n_components=None,
        tall=False,
    ):
        """
        Calculates a projection from each view to a latentent space such that
        the sum of pariwise latent space correlations is maximized. Each view
        'X' is normalized and the left singular vectors of 'X^T X' are
        calculated using SVD. The number of singular vectors kept is determined
        by either the percent variance explained, a given rank threshold, or a
        given number of components. The singular vectors kept are concatenated
        and SVD of that is taken and used to calculated projections for each
        view.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each sample will receive its own embedding.
        fraction_var : percent, default=0.9
            Explained variance for rank selection during initial SVD of each
            sample.
        sv_tolerance : float, optional, default=None
            Singular value threshold for rank selection during initial SVD of
            each sample.
        n_components : int (positive), optional, default=None
            Rank to truncate to during initial SVD of each sample.
        tall : boolean, default=False
            Set to true if n_samples > n_features, speeds up SVD

        Attributes
        ----------
        _projection_mats : list of arrays A projection matrix for each view,
            from the given space to the latent space self._ranks : list of ints
            number of left singular vectors kept for each view during the first
            SVD
        """
        # Project data to kernel
        Xs = check_Xs(Xs, multiview=True)
        data = [self._center(x) for x in Xs]
        
        for x in data:
            x = _make_kernel(x, ktype= 'gaussian', sigma=1,
                     degree=2)

        n = Xs[0].shape[0]
        min_m = min(X.shape[1] for X in Xs)
        

        Uall = []
        Sall = []
        Vall = []
        ranks = []

        for x in data:
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
            if sv_tolerance:
                if not isinstance(sv_tolerance, float) and not isinstance(
                    sv_tolerance, int
                ):
                    raise TypeError("sv_tolerance must be numeric")
                elif sv_tolerance <= 0:
                    raise ValueError("sv_tolerance must be greater than 0")

                rank = sum(s > sv_tolerance)
            elif n_components:
                if not isinstance(n_components, int):
                    raise TypeError("n_components must be an integer")
                elif n_components <= 0:
                    raise ValueError("n_components must be greater than 0")
                elif n_components > min((n, min_m)):
                    raise ValueError(
                        "n_components must be less than or equal to the \
                            minimum input rank"
                    )

                rank = n_components
            else:
                if not isinstance(fraction_var, float) and not isinstance(
                    fraction_var, int
                ):
                    raise TypeError("fraction_var must be an integer or float")
                elif fraction_var <= 0 or fraction_var > 1:
                    raise ValueError("fraction_var must be in (0,1]")

                s2 = np.square(s)
                rank = sum(np.cumsum(s2 / sum(s2)) < fraction_var) + 1
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
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            The data to embed based on the prior fit function. If
            view_idx defined, Xs is 2D, single view
        view_idx: int
            The index of the view whose projection to use on Xs.
            For transforming a single view inpu.

        Returns
        -------
        Xs_transformed : array-like 2D
            if view_idx not None, shape same as Xs
        """
        if self._projection_mats is None:
            raise RuntimeError("Must call fit function before transform")
        Xs = check_Xs(Xs)
        if view_idx is not None:
            return self._center(Xs[0]) @ self._projection_mats[view_idx]
        else:
            return np.array(
                [
                    self._center(x) @ proj
                    for x, proj in zip(Xs, self._projection_mats)
                ]
            )

    def fit_transform(self, Xs, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to Xs optional parameters fit_params and returns a
        transformed version of the Xs.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each sample will receive its own embedding.

        Returns
        -------
        Xs_transformed : array-like 2D if view_idx not None, otherwise
            (n_views, n_samples, n_components)
        """

        return self.fit(Xs, **fit_params).transform(Xs)

def _demean(d):
    """
    Calculates difference from mean of the data

    Parameters
    ----------
    d
        Data of interest (Array)

    Returns
    -------
    diff
        Difference from the mean (Array)
    """
    diff = d - d.mean(0)
    return diff

def _make_kernel(d, normalize=True, ktype="linear", sigma=1.0, degree=2):
    """
    Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = sigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree

    Parameters
    ----------
    d
        Data (Array)
    ktype
        Type of kernel if kernel is True. (String)
        Value can be 'linear' (default), 'gaussian' or 'polynomial'.
    sigma
        Parameter if the kernel is a Gaussian kernel. (Float)
    degree
        Parameter if the kernel is a Polynomial kernel. (Integer)

    Returns
    -------
    kernel
        Kernel that data is projected to (Array)
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == "linear":
        kernel = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform

        pairwise_dists = squareform(pdist(cd, "euclidean"))
        kernel = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    elif ktype == "poly":
        kernel = np.dot(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.0
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel