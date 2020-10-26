from scipy.linalg import eigh
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import svd_flip
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.utils import check_random_state


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix. Handles the case where
    X is either dense or sparse.

    Parameters
    ----------
    X: array-like,  shape (n, d)

    rank: int, None
        rank of the desired SVD. If None, is set to min(X.shape)

    Output
    ------
    U, D, V

    U: array-like, shape (n, rank)
        Orthonormal matrix of left singular vectors.

    D: list, shape (rank, )
        Singular values in non-increasing order (e.g. D[0] is the largest).

    V: array-like, shape (d, rank)
        Orthonormal matrix of right singular vectors

    """
    if rank is None:
        rank = min(X.shape)

    rank = int(rank)
    assert 1 <= rank and rank <= min(X.shape)

    if rank <= min(X.shape) - 1:
        U, D, V = svds(X, rank)
        U, D, V = sort_svds(U, D, V)

    else:
        assert not issparse(X)

        U, D, V = full_svd(X, full_matrices=False)
        V = V.T

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V[:, :rank]

    # enforce deterministic output
    U, V = svd_flip(U, V.T)
    V = V.T

    return U, D, V


def sort_svds(U, D, V):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function returns the singular values in decreasing order.

    Parameters
    ----------
    U, D, V : the outputs from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
        Input ordered by decreasing singular values
    """
    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V


def eigh_wrapper(A, B=None, rank=None, eval_descending=True):
    r"""
    Solves a symmetric eigenvector or genealized eigenvector problem.

        .. math:
            A v = \lambda v

    or

        .. math:
            A v = \labmda B v

    where A (and B) are symmetric (hermetian).

    Parameters
    ----------
    A: array-like, shape (n x n)
        The LHS matrix.

    B: None, array-like, shape (n x n)
        The (optional) RHS matrix.

    rank: None, int
        Number of components to compute.

    eval_descending: bool
        Whether or not to compute largest or smallest eigenvalues.
        If True, will compute largest rank eigenvalues and
        eigenvalues are returned in descending order. Otherwise,
        computes smallest eigenvalues and returns them in ascending order.

    Output
    ------
    evals : numpy.ndarray, shape (rank,)
        Solution eigenvalues

    evecs : numpy.ndarray, shape (n, rank)
        Solution eigenvectors
    """

    if rank is not None:
        n_max_evals = A.shape[0]

        if eval_descending:
            eigvals_idxs = (n_max_evals - rank, n_max_evals - 1)
        else:
            eigvals_idxs = (0, rank - 1)
    else:
        eigvals_idxs = None

    evals, evecs = eigh(a=A, b=B, subset_by_index=eigvals_idxs)

    if eval_descending:
        ev_reordering = np.argsort(-evals)
        evals = evals[ev_reordering]
        evecs = evecs[:, ev_reordering]

    evecs = svd_flip(evecs, evecs.T)[0]

    return evals, evecs


def rand_orthog(n, K, random_state=None):
    """
    Samples a random orthonormal matrix.

    Parameters
    ----------
    n : int, positive
        Number of rows in the matrix

    K : int, positive
        Number of columns in the matrix

    random_state : None | int | instance of RandomState, optional
        Seed to set randomization for reproducible results

    Returns
    -------
    A: array-like, (n, K)
        A random, column orthonormal matrix.

    Notes
    -----
    See Section A.1.1 of https://arxiv.org/pdf/0909.3052.pdf
    """
    rng = check_random_state(random_state)

    Z = rng.normal(size=(n, K))
    Q, R = np.linalg.qr(Z)

    s = np.ones(K)
    neg_mask = rng.uniform(size=K) > 0.5
    s[neg_mask] = -1

    return Q * s
