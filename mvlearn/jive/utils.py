import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd

from mvlearn.jive.interface import LinearOperator
from mvlearn.jive.convert2scipy import convert2scipy


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix. Handles the case where
    X is either dense or sparse.

    Parameters
    ----------
    X: array-like,  shape (N, D)

    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V

    U: array-like, shape (N, rank)
        Orthonormal matrix of left singular vectors.

    D: list, shape (rank, )
        Singular values in non-increasing order (e.g. D[0] is the largest).

    V: array-like, shape (D, rank)
        Orthonormal matrix of right singular vectors

    """
    full = False
    if rank is None or rank == min(X.shape):
        full = True

    if isinstance(X, LinearOperator):
        scipy_svds = svds(convert2scipy(X), rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    elif issparse(X) or not full:
        assert rank <= min(X.shape) - 1  # svds cannot compute the full svd
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    else:
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V[:, :rank]

    return U, D, V


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order

    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V


def centering(X, method="mean"):
    """
    Mean centers columns of a matrix.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The input matrix.

    method: str, None
        How to center.

    Output
    ------
    X_centered, center

    X_centered: array-like, shape (n_samples, n_features)
        The centered version of X whose columns have mean zero.

    center: array-like, shape (n_features, )
        The column means of X.
    """

    if type(method) == bool and method:
        method = "mean"

    if issparse(X):
        raise NotImplementedError
        # X_centered = MeanCentered(blocks[bn], centers_[bn])
    else:
        if method == "mean":
            center = np.array(X.mean(axis=0)).reshape(-1)
            X_centered = X - center
        else:
            center = None
            X_centered = X

    return X_centered, center
