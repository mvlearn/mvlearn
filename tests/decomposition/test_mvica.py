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
from mvlearn.decomposition.mv_ica import _hungarian
from mvlearn.decomposition import MultiviewICA, GroupICA, PermICA


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
    order, _ = _hungarian(M)
    return 1 - M[np.arange(M.shape[0]), order]


@pytest.mark.parametrize(
    ("algo, init"),
    [
        (PermICA, None),
        (GroupICA, None),
        (MultiviewICA, "permica"),
        (MultiviewICA, "groupica"),
    ],
)
def test_ica(algo, init):
    # Test that all algo can recover the sources
    sigma = 1e-4
    n, v, p, t = 3, 10, 5, 1000
    # Generate signals
    rng = np.random.RandomState(0)
    S_true = rng.laplace(size=(p, t))
    S_true = normalize(S_true)
    A_list = rng.randn(n, v, p)
    noises = rng.randn(n, v, t)
    X = np.array([A.dot(S_true) for A in A_list])
    X += [sigma * N for A, N in zip(A_list, noises)]
    # Run ICA
    if init is None:
        algo = algo(
            n_components=5,
            tol=1e-5,
        ).fit(np.swapaxes(X,1,2))
    else:
        algo = algo(
            n_components=5,
            tol=1e-5,
            init=init,
        ).fit(np.swapaxes(X,1,2))
    K = np.swapaxes(algo.components_, 1, 2)
    W = np.swapaxes(algo.unmixings_, 1, 2)
    S = algo.source_.T
    dist = np.mean([amari_d(W[i].dot(K[i]), A_list[i]) for i in range(n)])
    S = normalize(S)
    err = np.mean(error(np.abs(S.dot(S_true.T))))
    assert dist < 0.01
    assert err < 0.01