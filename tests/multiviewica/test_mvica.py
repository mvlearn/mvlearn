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

import pytest
import numpy as np
import scipy
from mvlearn.decomposition import MultiviewICA
from mvlearn.utils import requires_multiviewica


def hungarian(M):
    u, order = scipy.optimize.linear_sum_assignment(-abs(M))
    vals = M[u, order]
    return order, np.sign(vals)


def normalize(A):
    A_ = A - np.mean(A, axis=1, keepdims=True)
    A_ = A_ / np.linalg.norm(A_, axis=1, keepdims=True)
    return A_


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def error(M):
    order, _ = hungarian(M)
    return 1 - M[np.arange(M.shape[0]), order]


# Initialize data
@pytest.fixture(scope="module")
def Xs():
    np.random.seed(0)
    view1 = np.random.random((10, 9))
    view2 = np.random.random((10, 9))
    Xs = [view1, view2]
    return np.asarray(Xs)


@requires_multiviewica
@pytest.mark.parametrize(
    ("algo", "init"), [(MultiviewICA, "permica"), (MultiviewICA, "groupica"),],
)
def test_ica(algo, init):
    # Test that all algo can recover the sources
    sigma = 1e-4
    n, v, p, t = 4, 10, 5, 1000
    # Generate signals
    rng = np.random.RandomState(0)
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, v, p)
    noises = rng.randn(n, v, t)
    Xs = [A.dot(S_true) for A in A_list]
    Xs = [X + sigma * N for X, A, N in zip(Xs, A_list, noises)]
    # Run ICA
    if init is None:
        algo = algo(n_components=5, tol=1e-5, multiview_output=False).fit(
            np.swapaxes(Xs, 1, 2)
        )
    else:
        algo = algo(
            n_components=5, tol=1e-5, init=init, multiview_output=False
        ).fit(np.swapaxes(Xs, 1, 2))
    K = algo.pca_components_
    W = algo.components_
    S = algo.transform(np.swapaxes(Xs, 1, 2)).T
    dist = np.mean(
        [
            amari_d(W[i], np.linalg.pinv(K[i]).T.dot(A_list[i]))
            for i in range(n)
        ]
    )
    S = normalize(S)
    err = np.mean(error(np.abs(S.dot(S_true.T))))
    assert dist < 0.01
    assert err < 0.01


@requires_multiviewica
def test_transform(Xs):
    ica = MultiviewICA(n_components=2)
    with pytest.raises(ValueError):
        ica.transform(Xs)
    assert ica.fit_transform(Xs).shape == (Xs.shape[0], Xs.shape[1], 2)

    ica = MultiviewICA()
    assert ica.fit_transform(Xs).shape == Xs.shape


@requires_multiviewica
@pytest.mark.parametrize("multiview_output", [True, False])
def test_inverse_transform(Xs, multiview_output):
    ica = MultiviewICA(n_components=2, multiview_output=multiview_output)
    with pytest.raises(ValueError):
        ica.inverse_transform(Xs)
    S = ica.fit_transform(Xs)
    Xs_mixed = ica.inverse_transform(S)
    avg_mixed = np.mean(
        [X @ np.linalg.pinv(C) for X, C in zip(Xs, ica.pca_components_)],
        axis=0,
    )
    avg_mixed2 = np.mean(
        [X @ np.linalg.pinv(C) for X, C in zip(Xs_mixed, ica.pca_components_)],
        axis=0,
    )
    assert np.linalg.norm(avg_mixed2 - avg_mixed) < 0.2


@requires_multiviewica
@pytest.mark.parametrize("multiview_output", [True, False])
@pytest.mark.parametrize("index", [[0, 1], [1, 2], [0, 2], [0, 1, 2], None])
@pytest.mark.parametrize(
    "inverse_index", [[0, 1], [1, 2], [0, 2], [0, 1, 2], None]
)
def test_index(index, inverse_index, multiview_output):
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    X2 = np.copy(X)
    X2[:, 1] += rng.randn(n) * 0.00001
    X2 = X2.dot(rng.rand(p, p))

    X3 = np.copy(X)
    X3[:, 1] += rng.randn(n) * 0.00001
    X3 = X3.dot(rng.rand(p, p))

    Xs = [X, X2, X3]
    ica = MultiviewICA(
        n_components=2, init="groupica", multiview_output=multiview_output,
    ).fit(Xs)
    if index is not None:
        Xs_transform = [Xs[i] for i in index]
        len_index = len(index)
    else:
        len_index = 3
        Xs_transform = np.copy(Xs)

    if inverse_index is not None:
        Xs_inverse = [Xs[i] for i in inverse_index]
        len_inverse_index = len(inverse_index)
    else:
        len_inverse_index = 3
        Xs_inverse = np.copy(Xs)

    Y = ica.transform(Xs_transform, index=index)
    Y_inverse = ica.inverse_transform(Y, index=inverse_index)
    for X, X_estimated in zip(Xs_inverse, Y_inverse):
        np.testing.assert_allclose(X, X_estimated, atol=1e-3)


@requires_multiviewica
def test_inverse_transform_no_preproc(Xs):
    ica = MultiviewICA()
    S = ica.fit_transform(Xs)
    Xs_mixed = ica.inverse_transform(S)
    assert np.mean((Xs_mixed - Xs) ** 2) / np.mean(Xs ** 2) < 0.05


@requires_multiviewica
def test_fit_errors(Xs):
    with pytest.raises(ValueError):
        ica = MultiviewICA()
        ica.fit(Xs[:, :5, :])
    with pytest.raises(ValueError):
        ica = MultiviewICA(init="WRONG")
        ica.fit(Xs)
    with pytest.raises(TypeError):
        ica = MultiviewICA(init=list())
        ica.fit(Xs)


@requires_multiviewica
def test_fit(Xs, capfd):
    ica = MultiviewICA(verbose=True)
    ica.fit(Xs)
    out, err = capfd.readouterr()
    assert out[:2] == "it"
