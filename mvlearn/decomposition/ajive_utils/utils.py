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
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd


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

    if issparse(X) or not full:
        assert rank <= min(X.shape) - 1  # svds cannot compute the full svd
        U, D, V  = svds(X, rank)

        # Sort in decreasing order
        sv_reordering = np.argsort(-D)
        U = U[:, sv_reordering]
        D = D[sv_reordering]
        V = V.T[:, sv_reordering]

    else:
        U, D, V = full_svd(X, full_matrices=False)

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V.T[:, :rank]

    return U, D, V


def _centering(X, method="mean"):
    r"""
    Mean centers columns of a matrix.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input matrix.

    method : str
        Centering method

    Returns
    -------
    X_centered : array-like, shape (n_samples, n_features)
        The centered X

    center : array-like, shape (n_features, )
        The column means of X.
    """

    if type(method) == bool and method:
        method = "mean"

    if issparse(X):
        raise NotImplementedError
    else:
        if method == "mean":
            center = np.array(X.mean(axis=0)).reshape(-1)
            X_centered = X - center
        else:
            center = None
            X_centered = X

    return X_centered, center
