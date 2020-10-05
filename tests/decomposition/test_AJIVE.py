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
from numpy.testing import assert_equal, assert_allclose
import pytest
from mvlearn.decomposition import AJIVE
from scipy.linalg import orth


"""
INITIALIZATION
"""


@pytest.fixture(scope="module")
def data():

    np.random.seed(12)

    # First View
    V1_joint = np.vstack([-1 * np.ones((10, 20)), np.ones((10, 20))])

    V1_joint = np.hstack([np.zeros((20, 80)), V1_joint])

    V1_indiv_t = np.vstack(
        [
            np.ones((4, 50)),
            -1 * np.ones((4, 50)),
            np.zeros((4, 50)),
            np.ones((4, 50)),
            -1 * np.ones((4, 50)),
        ]
    )

    V1_indiv_b = np.vstack(
        [np.ones((5, 50)), -1 * np.ones((10, 50)), np.ones((5, 50))]
    )

    V1_indiv_tot = np.hstack([V1_indiv_t, V1_indiv_b])

    V1_noise = np.random.normal(loc=0, scale=1, size=(20, 100))

    # Second View
    V2_joint = np.vstack([np.ones((10, 10)), -1 * np.ones((10, 10))])

    V2_joint = 5000 * np.hstack([V2_joint, np.zeros((20, 10))])

    V2_indiv = 5000 * np.vstack(
        [
            -1 * np.ones((5, 20)),
            np.ones((5, 20)),
            -1 * np.ones((5, 20)),
            np.ones((5, 20)),
        ]
    )

    V2_noise = 5000 * np.random.normal(loc=0, scale=1, size=(20, 20))

    # View Construction

    V1 = V1_indiv_tot + V1_joint + V1_noise

    V2 = V2_indiv + V2_joint + V2_noise

    Views_Same = [V1, V1]
    Views_Different = [V1, V2]

    return {
        "same_views": Views_Same,
        "diff_views": Views_Different,
    }


@pytest.fixture(scope='module')
def ajive(data):
    aj = AJIVE(init_signal_ranks=[2, 3])
    joint_mats = aj.fit_transform(Xs=data['diff_views'])
    aj.joint_mats = joint_mats

    return aj


"""
TESTS
"""


def test_correct_estimates(ajive):
    """
    Check AJIVE found correct rank estimates
    """
    assert_equal(ajive.joint_rank_, 1)
    assert_equal(ajive.individual_ranks_, [1, 3])


def test_joint_SVDs(ajive):
    U = ajive.joint_scores_
    rank = ajive.joint_rank_
    n = ajive.view_shapes_[0][0]

    assert_equal(U.shape, (n, rank))
    assert_allclose(np.dot(U.T, U), np.eye(rank))


def test_indiv_SVDs(ajive):
    """
    Check each block specific SVD
    """
    Us = ajive.individual_scores_
    Ds = ajive.individual_svals_
    Vs = ajive.individual_loadings_
    ranks = ajive.individual_ranks_
    shapes = ajive.view_shapes_

    for U, D, V, rank, (n, d) in zip(Us, Ds, Vs, ranks, shapes):
        assert_equal(U.shape, (n, rank))
        assert_allclose(np.dot(U.T, U), np.eye(rank), atol=1e-7)
        assert_equal(D.shape, (rank,))
        svals_nonincreasing = True
        for i in range(len(D) - 1):
            if D[i] < D[i + 1]:
                svals_nonincreasing = False
        assert svals_nonincreasing
        assert_equal(V.shape, (d, rank))
        assert_allclose(np.dot(V.T, V), np.eye(rank), atol=1e-7)


def test_centering(data, ajive):
    """
    check X_centered = I + J
    """
    Xs = data['diff_views']
    col_means = [np.mean(X, axis=0) for X in Xs]
    assert_equal(col_means, ajive.means_)


def test_same_joint_indiv_length(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2])
    Js = ajive.fit_transform(Xs=dat)
    Is = ajive.individual_mats_
    assert_equal(Js[0].shape, Is[0].shape)


@pytest.mark.parametrize("init_signal_ranks", [None, [2, 2]])
@pytest.mark.parametrize("joint_rank", [None, 0, 1])
@pytest.mark.parametrize("individual_ranks", [None, [0, 0], [1, 1]])
def test_same_joint(data, init_signal_ranks, individual_ranks, joint_rank):
    # Test same joint input result across varying inputs
    dat = data["same_views"]
    ajive = AJIVE(
        init_signal_ranks=init_signal_ranks,
        joint_rank=joint_rank,
        individual_ranks=individual_ranks
        )
    Js = ajive.fit_transform(Xs=dat)
    for i in np.arange(20):
        j = np.sum(Js[0][i] == Js[1][i])
        assert_equal(j, 100)


@pytest.mark.parametrize("init_signal_ranks", [None, [2, 2]])
@pytest.mark.parametrize("joint_rank", [None, 0, 4])
@pytest.mark.parametrize("individual_ranks", [None, [0, 0], [1, 1]])
def test_same_indiv(data, init_signal_ranks, individual_ranks, joint_rank):
    # Test same indiv input result across varying inputs
    dat = data["same_views"]
    ajive = AJIVE(
        init_signal_ranks=init_signal_ranks,
        joint_rank=joint_rank,
        individual_ranks=individual_ranks
        )
    ajive = ajive.fit(Xs=dat)
    Is = ajive.individual_mats_
    assert_allclose(Is[0], Is[1])


# Test 0 ranks
def test_indiv_rank_0(data):
    dat = data["diff_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], individual_ranks=[0, 0])
    _ = ajive.fit_transform(dat)
    Is = ajive.individual_mats_
    assert_allclose(Is[0], 0)
    assert_allclose(Is[1], 0)


def test_joint_rank_0(data):
    dat = data["diff_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=0)
    Js = ajive.fit_transform(dat)
    assert_allclose(Js[0], 0)
    assert_allclose(Js[1], 0)


def test_indiv_rank(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], individual_ranks=[2, 1])
    ajive = ajive.fit(Xs=dat)
    assert_equal(ajive.individual_ranks_[0], 2)


def test_joint_rank(data):
    dat = data["same_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=2)
    ajive = ajive.fit(Xs=dat)
    assert_equal(ajive.joint_rank, 2)


def test_fit_elbows():
    n = 10
    elbows = 3
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

    assert_equal(ajive.init_signal_ranks_[0], 4)


def test_random_state(data):
    # Tests reproducible simulations
    dat = data["same_views"]
    ajive1 = AJIVE(init_signal_ranks=[2, 2], random_state=0)
    ajive1 = ajive1.fit(Xs=dat)
    ajive2 = AJIVE(init_signal_ranks=[2, 2], random_state=0)
    ajive2 = ajive2.fit(Xs=dat)
    assert_allclose(ajive1.wedin_samples_, ajive2.wedin_samples_)
    assert_allclose(ajive1.random_sv_samples_, ajive2.random_sv_samples_)


"""
TESTS (Errors, warnings)
"""


# Sees whether incorrect signals will work
def test_wrong_signal_ranks(data):
    # rank < 0
    dat = data["diff_views"]
    ajive = AJIVE(init_signal_ranks=[-1, -4])
    with pytest.raises(ValueError):
        ajive.fit(Xs=dat)
    ajive = AJIVE(init_signal_ranks=[max(dat[0].shape)+1, -4])
    with pytest.raises(ValueError):
        ajive.fit(Xs=dat)


def test_check_joint_rank_large(data):
    # Joint rank < sum(init_signal_ranks)
    with pytest.raises(ValueError):
        dat = data["same_views"]
        ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=5)
        ajive = ajive.fit(Xs=dat)


def test_zero_rank_warn(data):
    # warn returing rank 0 joint
    dat = data["diff_views"]
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=0)
    with pytest.warns(RuntimeWarning):
        _ = ajive.fit_transform(dat)


def test_signal_ranks_None(data):
    # Both init rank inputs are None
    dat = data["same_views"]
    with pytest.raises(ValueError):
        ajive = AJIVE(init_signal_ranks=None, n_elbows=None)
        ajive = ajive.fit(Xs=dat)


@pytest.mark.parametrize("n_wedin_samples", [None, 0, -1])
@pytest.mark.parametrize("n_randdir_samples", [None, 0, -1])
def test_invalid_samples(data, n_wedin_samples, n_randdir_samples):
    # invalid number of samples
    dat = data["same_views"]
    with pytest.raises(ValueError):
        ajive = AJIVE(n_wedin_samples=n_wedin_samples)
        ajive = ajive.fit(Xs=dat)
    with pytest.raises(ValueError):
        ajive = AJIVE(n_randdir_samples=n_randdir_samples)
        ajive = ajive.fit(Xs=dat)


@pytest.mark.parametrize("init_signal_ranks", [dict, 0, [1], [1, 2, 3]])
@pytest.mark.parametrize("individual_ranks", [dict, 0, [1], [1, 2, 3]])
def test_invalid_ranks(data, init_signal_ranks, individual_ranks):
    # invalid number of samples
    dat = data["same_views"]
    with pytest.raises(ValueError):
        ajive = AJIVE(init_signal_ranks=init_signal_ranks)
        ajive = ajive.fit(Xs=dat)
    with pytest.raises(ValueError):
        ajive = AJIVE(individual_ranks=individual_ranks)
        ajive = ajive.fit(Xs=dat)


def test_signal_rank_warning():
    # Throws warning signal rank is larger than possible rank, sets to max-1
    ajive = AJIVE(init_signal_ranks=[2, 2], joint_rank=2)
    a = np.vstack([[1, 1], [1, 1]])
    with pytest.warns(RuntimeWarning):
        ajive.fit([a, a])

    assert ajive.init_signal_ranks_ == [1, 1]
