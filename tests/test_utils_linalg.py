import numpy as np
from numpy.testing import assert_almost_equal
from mvlearn.utils import rand_orthog, eigh_wrapper, sort_svds, svd_wrapper


def test_rand_orthog():
    X = rand_orthog(10, 10)
    assert_almost_equal(X @ X.T, np.eye(10))


def test_eigh_wrapper():
    np.random.seed(0)
    A = np.random.standard_normal((5, 5))
    A = A @ A.T
    B = np.random.standard_normal((5, 5))
    B = B @ B.T

    evals, evecs = eigh_wrapper(A, B, rank=5, eval_descending=True)
    assert np.all(np.diff(evals) <= 0)
    assert evecs.shape[1] == 5

    evals, evecs = eigh_wrapper(A, B, rank=4, eval_descending=False)
    assert np.all(np.diff(evals) >= 0)
    assert evecs.shape[1] == 4

    evecs, evals, _ = sort_svds(evecs, evals, evecs.T)
    assert np.all(np.diff(evals) <= 0)
    assert evecs.shape[1] == 4


def test_svds_wrapper():
    np.random.seed(0)
    A = np.random.standard_normal((10, 5))
    U, D, V = svd_wrapper(A)
    assert_almost_equal(U.T @ U, np.eye(5))
    assert_almost_equal(V.T @ V, np.eye(5))
    assert np.all(np.diff(D) <= 0)

    U, D, V = svd_wrapper(A, rank=2)
    assert_almost_equal(U.T @ U, np.eye(2))
    assert_almost_equal(V.T @ V, np.eye(2))
    assert np.all(np.diff(D) <= 0)
