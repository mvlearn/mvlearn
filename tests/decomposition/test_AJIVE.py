import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pandas as pd
import pytest
from mvlearn.decomposition.ajive import (
    AJIVE,
    ajive_full_estimate_heatmaps,
    data_block_heatmaps,
)
from mvlearn.decomposition.ajive_utils.utils import svd_wrapper
from scipy.sparse import csr_matrix
from scipy.linalg import orth
from pandas.testing import assert_frame_equal, assert_series_equal


class TestFig2Runs(unittest.TestCase):
    @classmethod
    def setUp(self):

        np.random.seed(12)

        # First View
        V1_joint = np.bmat([[-1 * np.ones((10, 20))], [np.ones((10, 20))]])

        V1_joint = np.bmat([np.zeros((20, 80)), V1_joint])

        V1_indiv_t = np.bmat(
            [
                [np.ones((4, 50))],
                [-1 * np.ones((4, 50))],
                [np.zeros((4, 50))],
                [np.ones((4, 50))],
                [-1 * np.ones((4, 50))],
            ]
        )

        V1_indiv_b = np.bmat(
            [[np.ones((5, 50))], [-1 * np.ones((10, 50))], [np.ones((5, 50))]]
        )

        V1_indiv_tot = np.bmat([V1_indiv_t, V1_indiv_b])

        V1_noise = np.random.normal(loc=0, scale=1, size=(20, 100))

        # Second View
        V2_joint = np.bmat([[np.ones((10, 10))], [-1 * np.ones((10, 10))]])

        V2_joint = 5000 * np.bmat([V2_joint, np.zeros((20, 10))])

        V2_indiv = 5000 * np.bmat(
            [
                [-1 * np.ones((5, 20))],
                [np.ones((5, 20))],
                [-1 * np.ones((5, 20))],
                [np.ones((5, 20))],
            ]
        )

        V2_noise = 5000 * np.random.normal(loc=0, scale=1, size=(20, 20))

        # View Construction

        X = V1_indiv_tot + V1_joint + V1_noise

        Y = V2_indiv + V2_joint + V2_noise

        obs_names = ["sample_{}".format(i) for i in range(X.shape[0])]
        var_names = {
            "x": ["x_var_{}".format(i) for i in range(X.shape[1])],
            "y": ["y_var_{}".format(i) for i in range(Y.shape[1])],
        }

        X = pd.DataFrame(X, index=obs_names, columns=var_names["x"])
        Y = pd.DataFrame(Y, index=obs_names, columns=var_names["y"])

        self.ajive = AJIVE(init_signal_ranks=[2, 3]).fit(
            Xs=[X, Y], view_names=["x", "y"]
        )

        self.X = X
        self.Y = Y
        self.obs_names = obs_names
        self.var_names = var_names

    def test_has_attributes(self):
        """
        Check AJIVE has important attributes
        """
        self.assertTrue(hasattr(self.ajive, "blocks_"))
        self.assertTrue(hasattr(self.ajive, "common_"))
        self.assertTrue(hasattr(self.ajive.blocks_["x"], "joint"))
        self.assertTrue(hasattr(self.ajive.blocks_["x"], "individual"))
        self.assertTrue(hasattr(self.ajive.blocks_["y"], "joint"))
        self.assertTrue(hasattr(self.ajive.blocks_["y"], "individual"))

    def test_correct_estimates(self):
        """
        Check AJIVE found correct rank estimates
        """
        self.assertEqual(self.ajive.common_.rank, 1)
        self.assertEqual(self.ajive.blocks_["x"].individual.rank, 1)
        self.assertEqual(self.ajive.blocks_["y"].individual.rank, 3)

    def test_matrix_decomposition(self):
        """
        check X_centered = I + J + E
        """
        X_cent = self.X - self.X.mean(axis=0)
        Rx = np.array(X_cent) - (
            self.ajive.blocks_["x"].joint.full_
            + self.ajive.blocks_["x"].individual.full_
            + self.ajive.blocks_["x"].noise_
        )

        self.assertTrue(np.allclose(Rx, 0))

        Y_cent = self.Y - self.Y.mean(axis=0)
        Ry = np.array(Y_cent) - (
            self.ajive.blocks_["y"].joint.full_
            + self.ajive.blocks_["y"].individual.full_
            + self.ajive.blocks_["y"].noise_
        )

        self.assertTrue(np.allclose(Ry, 0))

    def test_common_SVD(self):
        """
        Check common SVD
        """
        U, D, V = self.ajive.common_.get_UDV()
        rank = self.ajive.common_.rank
        n = self.X.shape[0]
        d = sum(self.ajive.init_signal_ranks_.values())
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

    def test_block_specific_SVDs(self):
        """
        Check each block specific SVD
        """
        U, D, V = self.ajive.blocks_["x"].joint.get_UDV()
        rank = 1
        n, d = self.X.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

        U, D, V = self.ajive.blocks_["x"].individual.get_UDV()
        rank = 1
        n, d = self.X.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

        U, D, V = self.ajive.blocks_["y"].joint.get_UDV()
        rank = 1
        n, d = self.Y.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

    def test_list_input(self):
        """
        Check AJIVE can take a list input.
        """
        ajive = AJIVE(init_signal_ranks=[2, 3])
        ajive.fit(Xs=[self.X, self.Y])
        self.assertTrue(set(ajive.block_names) == set([0, 1]))

    def test_dont_store_full(self):
        """
        Make sure setting store_full = False works
        """
        ajive = AJIVE(init_signal_ranks=[2, 3], store_full=False)
        ajive.fit(Xs=[self.X, self.Y])

        self.assertTrue(ajive.blocks_[0].joint.full_ is None)
        self.assertTrue(ajive.blocks_[0].individual.full_ is None)
        self.assertTrue(ajive.blocks_[1].joint.full_ is None)
        self.assertTrue(ajive.blocks_[1].individual.full_ is None)

    def test_rank0(self):
        """
        Check setting joint/individual rank to zero works
        """
        ajive = AJIVE(init_signal_ranks=[2, 3], joint_rank=0)
        ajive.fit(Xs=[self.X, self.Y])
        self.assertTrue(ajive.common_.rank == 0)
        self.assertTrue(ajive.blocks_[0].joint.rank == 0)
        self.assertTrue(ajive.blocks_[0].joint.scores_ is None)

        ajive = AJIVE(init_signal_ranks=[2, 3], indiv_ranks=[0, 1])
        ajive.fit(Xs=[self.X, self.Y])
        self.assertTrue(ajive.blocks_[0].individual.rank == 0)
        self.assertTrue(ajive.blocks_[0].individual.scores_ is None)

    def test_centering(self):
        xmean = self.X.mean(axis=0)
        ymean = self.Y.mean(axis=0)

        self.assertTrue(np.allclose(self.ajive.centers_["x"], xmean))
        self.assertTrue(np.allclose(self.ajive.blocks_["x"].joint.m_, xmean))
        self.assertTrue(
            np.allclose(self.ajive.blocks_["x"].individual.m_, xmean)
        )

        self.assertTrue(np.allclose(self.ajive.centers_["y"], ymean))
        self.assertTrue(np.allclose(self.ajive.blocks_["y"].joint.m_, ymean))
        self.assertTrue(
            np.allclose(self.ajive.blocks_["y"].individual.m_, ymean)
        )

        # no centering
        ajive = AJIVE(init_signal_ranks=[2, 3], center=False)
        ajive = ajive.fit(Xs=[self.X, self.Y], view_names=["x", "y"])
        self.assertTrue(ajive.centers_["x"] is None)
        self.assertTrue(ajive.centers_["y"] is None)

        # only center x
        ajive = AJIVE(init_signal_ranks=[2, 3], center=[True, False])
        ajive = ajive.fit(Xs=[self.X, self.Y], view_names=["x", "y"])
        self.assertTrue(np.allclose(ajive.centers_["x"], xmean))
        self.assertTrue(ajive.centers_["y"] is None)


if __name__ == "__main__":
    unittest.main()


def svd_checker(U, D, V, n, d, rank):
    checks = {}

    # scores shape
    checks["scores_shape"] = U.shape == (n, rank)

    # scores have orthonormal columns
    checks["scores_ortho"] = np.allclose(np.dot(U.T, U), np.eye(rank))

    # singular values shape
    checks["svals_shape"] = D.shape == (rank,)

    # singular values are in non-increasing order
    svals_nonincreasing = True
    for i in range(len(D) - 1):
        if D[i] < D[i + 1]:
            svals_nonincreasing = False
    checks["svals_nonincreasing"] = svals_nonincreasing

    # loadings shape
    checks["loading_shape"] = V.shape == (d, rank)

    # loadings have orthonormal columns
    checks["loadings_ortho"] = np.allclose(np.dot(V.T, V), np.eye(rank))

    return checks


"""
DATA INITIALIZATION
"""


@pytest.fixture(scope="module")
def data():

    np.random.seed(12)

    # First View
    V1_joint = np.bmat([[-1 * np.ones((10, 20))], [np.ones((10, 20))]])

    V1_joint = np.bmat([np.zeros((20, 80)), V1_joint])

    V1_indiv_t = np.bmat(
        [
            [np.ones((4, 50))],
            [-1 * np.ones((4, 50))],
            [np.zeros((4, 50))],
            [np.ones((4, 50))],
            [-1 * np.ones((4, 50))],
        ]
    )

    V1_indiv_b = np.bmat(
        [[np.ones((5, 50))], [-1 * np.ones((10, 50))], [np.ones((5, 50))]]
    )

    V1_indiv_tot = np.bmat([V1_indiv_t, V1_indiv_b])

    V1_noise = np.random.normal(loc=0, scale=1, size=(20, 100))

    # Second View
    V2_joint = np.bmat([[np.ones((10, 10))], [-1 * np.ones((10, 10))]])

    V2_joint = 5000 * np.bmat([V2_joint, np.zeros((20, 10))])

    V2_indiv = 5000 * np.bmat(
        [
            [-1 * np.ones((5, 20))],
            [np.ones((5, 20))],
            [-1 * np.ones((5, 20))],
            [np.ones((5, 20))],
        ]
    )

    V2_noise = 5000 * np.random.normal(loc=0, scale=1, size=(20, 20))

    # View Construction

    V1 = V1_indiv_tot + V1_joint + V1_noise

    V2 = V2_indiv + V2_joint + V2_noise

    # Creating Sparse views
    V1_sparse = np.array(np.zeros_like(V1))
    V2_sparse = np.array(np.zeros_like(V2))
    V1_sparse[0, 0] = 1
    V2_sparse[0, 0] = 3
    V1_Bad = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    V2_Bad = csr_matrix([[1, 2, 3], [7, 0, 3], [1, 2, 2]])

    Views_Same = [V1, V1]
    Views_Different = [V1, V2]
    Views_Sparse = [V1_sparse, V2_sparse]
    Views_Bad = [V1_Bad, V2_Bad]

    return {
        "same_views": Views_Same,
        "diff_views": Views_Different,
        "sparse_views": Views_Sparse,
        "bad_views": Views_Bad,
    }


"""
TESTS
"""


def test_joint_indiv_length(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    assert blocks[0]["joint"].shape == blocks[0]["individual"].shape


def test_joint_noise_length(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    assert blocks[0]["joint"].shape == blocks[0]["noise"].shape


def test_joint(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    for i in np.arange(100):
        j = np.sum(blocks[0]["joint"][i] == blocks[1]["joint"][i])
        assert j == 20


def test_indiv(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    for i in np.arange(100):
        j = np.sum(blocks[0]["individual"][i] == blocks[1]["individual"][i])
        assert j == 20


# Sees whether incorrect signals will work
def test_wrong_sig(data):
    dat = data["diff_views"]
    ajive = AJIVE(init_signal_ranks=[-1, -4])
    try:
        ajive.fit(Xs=dat)
        j = 0
    except:
        j = 1
    assert j == 1


def test_check_sparse(data):
    dat = data["sparse_views"]
    spar_mat = dat[0]
    assert np.sum(spar_mat == 0) > np.sum(spar_mat != 0)
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    assert np.sum(np.sum(blocks[0]["individual"] == 0)) > np.sum(
        np.sum(blocks[0]["individual"] != 0)
    )


# Check valueerror for general linear operators
def test_check_gen_lin_op_scipy(data):
    with pytest.raises(TypeError):
        dat = data["bad_views"]
        ajive = AJIVE(init_signal_ranks=[2, 2])
        ajive.fit(Xs=dat)


def test_get_ranks_not_computed(data):
    with pytest.raises(ValueError):
        ajive = AJIVE(init_signal_ranks=[2, 2])
        ajive.get_ranks()


def test_check_joint_rank_large(data):
    with pytest.raises(ValueError):
        dat = data["same_views"]
        ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=5)
        ajive.fit(Xs=dat)


def test_indiv_rank(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], indiv_ranks=[2, 1])
    ajive.fit(Xs=dat)
    assert ajive.indiv_ranks[0] == 2


def test_joint_rank(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=2)
    ajive.fit(Xs=dat)
    assert ajive.joint_rank == 2


def test_is_fit():
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=2)
    assert ajive.is_fit_ == False


def test_n_randdir():
    ajive = AJIVE(init_signal_ranks=[2, 2], n_randdir_samples=5)
    assert ajive.n_randdir_samples == 5


def test_n_wedin():
    ajive = AJIVE(init_signal_ranks=[2, 2], n_wedin_samples=6)
    assert ajive.n_wedin_samples == 6


def test_precomp_init_svd(data):
    dat = data["same_views"]
    precomp = []
    for i in dat:
        precomp.append(svd_wrapper(i))
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=1)
    ajive.fit(dat, precomp_init_svd=precomp)
    p = 3
    assert p == 3

def test_block_names_not_fit():
    ajive = AJIVE()
    assert ajive.block_names is None

def test__repr__(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])

    assert ajive.__repr__() == "No data has been fitted yet"

    ajive.fit(Xs=dat)
    blocks = ajive.transform(return_dict=True)
    r = "joint rank: {}".format(ajive.common_.rank)
    for bn in ajive.block_names:
                indiv_rank = ajive.blocks_[bn].individual.rank
                r += ", block {} indiv rank: {}".format(bn, indiv_rank)
    assert ajive.__repr__() == r
    
def test_results_dict(data):
    dat = data["same_views"]
    precomp = []
    for i in dat:
        precomp.append(svd_wrapper(i))
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=1)
    ajive.fit(dat, precomp_init_svd=precomp)
    results = ajive.results_dict()
    assert_frame_equal(results['common']['scores'], ajive.common_.scores_)
    assert_series_equal(results['common']['svals'], ajive.common_.svals_)
    assert_frame_equal(results['common']['loadings'], ajive.common_.loadings_)
    assert_equal(results['common']['rank'], ajive.common_.rank)

    for bn in ajive.block_names:
        joint = ajive.blocks_[bn].joint
        indiv = ajive.blocks_[bn].individual
        assert_frame_equal(results[bn]['joint']['scores'], joint.scores_)
        assert_series_equal(results[bn]['joint']['svals'], joint.svals_)
        assert_frame_equal(results[bn]['joint']['loadings'], joint.loadings_)
        assert_equal(results[bn]['joint']['rank'], joint.rank)
        assert_frame_equal(results[bn]['joint']['full'], joint.full_)

        assert_frame_equal(results[bn]['individual']['scores'], indiv.scores_)
        assert_series_equal(results[bn]['individual']['svals'], indiv.svals_)
        assert_frame_equal(results[bn]['individual']['loadings'], indiv.loadings_)
        assert_equal(results[bn]['individual']['rank'], indiv.rank)
        assert_frame_equal(results[bn]['individual']['full'], indiv.full_)

        assert_frame_equal(results[bn]['noise'], ajive.blocks_[bn].noise_)

def test_get_ranks(data):
    dat = data["same_views"]
    precomp = []
    for i in dat:
        precomp.append(svd_wrapper(i))
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=1)
    ajive.fit(dat, precomp_init_svd=precomp)
    joint_rank, indiv_ranks = ajive.get_ranks()
    assert joint_rank == 1
    for rank1, rank2 in zip(indiv_ranks, [0, 1]):
        assert rank1 == rank2


# Plotting


def test_plot_diag(data):
    x = data["same_views"]
    data_block_heatmaps(x)
    p = 1
    assert p == 1


def test_ajive_plot(data):
    x = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=x)
    blocks = ajive.transform(return_dict=True)
    ajive_full_estimate_heatmaps(x, blocks)
    p = 1
    assert p == 1


def test_ajive_plot_list(data):
    x = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=x)
    blocks = ajive.transform(return_dict=False)
    ajive_full_estimate_heatmaps(x, blocks, names=["x1", "x2"])
    p = 1
    assert p == 1


def test_name_values(data):
    with pytest.raises(ValueError):
        x = data["same_views"]
        ajive = AJIVE(init_signal_ranks=[2, 2])
        ajive.fit(Xs=x, view_names=["1", "2", "3"])


def test_name_values_type(data):
    with pytest.raises(ValueError):
        x = data["same_views"]
        ajive = AJIVE(init_signal_ranks=[2, 2])
        ajive.fit(Xs=x, view_names={"jon": "first", "rich": "second"})


def test_traditional_output(data):
    x = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    ajive.fit(Xs=x, view_names=["x", "y"])
    ajive.transform(return_dict=False)

def test_fit_elbows():
    n=10; elbows=3
    np.random.seed(1)
    x = np.random.binomial(1, 0.6, (n ** 2)).reshape(n, n)
    xorth = orth(x)
    d = np.zeros(xorth.shape[0])
    for i in range(0, len(d), int(len(d) / (elbows + 1))):
        d[:i] += 10
    X = xorth.T.dot(np.diag(d)).dot(xorth)

    Xs = [X, X]

    ajive = AJIVE(n_elbows=2)
    ajive = ajive.fit(Xs)

    np.testing.assert_equal(list(ajive.init_signal_ranks_.values())[0], 4)