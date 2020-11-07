# MIT License

# Copyright (c) [2017] [Pierre Ablin and Hugo Richard]

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

import pytest

import numpy as np

from sklearn.utils._testing import assert_allclose
from mvlearn.decomposition import GroupICA


def amari_d(W, A=None):
    if A is None:
        P = W
    else:
        P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


def generate_signals(n_samples, n_sources, n_features, noise_level, rng):
    sources = rng.laplace(size=(n_samples, n_sources))
    mixings = [rng.randn(n_feature, n_sources) for n_feature in n_features]
    noises = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    Xs = [
        np.dot(sources, mixing.T) + noise_level * noise
        for mixing, noise in zip(mixings, noises)
    ]
    return Xs, sources, mixings


@pytest.mark.parametrize("n_components", [None, 1, 3])
@pytest.mark.parametrize(
    "n_individual_components", ["auto", None, 3, [2, 3, 4]]
)
@pytest.mark.parametrize("multiview_output", [True, False])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("solver", ["picard", "fastica"])
def test_transform(
    n_components, n_individual_components, multiview_output, solver, whiten
):
    ica_kwargs = dict(tol=1, whiten=whiten)
    ica = GroupICA(
        n_components=n_components,
        n_individual_components=n_individual_components,
        multiview_output=multiview_output,
        solver=solver,
        ica_kwargs=ica_kwargs,
    )
    rng = np.random.RandomState(0)
    n_samples = 100
    Xs, _, _ = generate_signals(n_samples, 2, [4, 5, 6], 0.1, rng)
    # check the shape of fit.transform
    X_r = ica.fit(Xs).transform(Xs)
    if multiview_output:
        for X in X_r:
            assert X.shape[0] == n_samples
            if n_components is not None:
                assert X.shape[1] == n_components
    else:
        assert X_r.shape[0] == n_samples
        if n_components is not None:
            assert X_r.shape[1] == n_components
    X_r2 = ica.transform(Xs)
    if multiview_output:
        for X, X2 in zip(X_r, X_r2):
            assert_allclose(X, X2)
    else:
        assert_allclose(X_r, X_r2)
    X_r = ica.fit_transform(Xs)
    X_r2 = ica.transform(Xs)
    if multiview_output:
        for X, X2 in zip(X_r, X_r2):
            assert_allclose(X, X2)
    else:
        assert_allclose(X_r, X_r2)


@pytest.mark.parametrize("n_features", [(4, 4, 4), (3, 4, 5)])
@pytest.mark.parametrize("n_components", [None, 2, 3])
@pytest.mark.parametrize("n_individual_components", ["auto", (2, 3, 2)])
@pytest.mark.parametrize("multiview_output", [True, False])
@pytest.mark.parametrize("prewhiten", [True, False])
@pytest.mark.parametrize("solver", ["picard", "fastica"])
def test_dimensions(
    n_features,
    n_components,
    n_individual_components,
    multiview_output,
    prewhiten,
    solver,
):
    n_samples = 10
    n_sources = 3
    noise_level = 0.1
    rng = np.random.RandomState(0)
    Xs, sources, mixings = generate_signals(
        n_samples, n_sources, n_features, noise_level, rng
    )
    ica_kwargs = dict(tol=1)
    ica = GroupICA(
        n_components=n_components,
        n_individual_components=n_individual_components,
        multiview_output=multiview_output,
        prewhiten=prewhiten,
        solver=solver,
        ica_kwargs=ica_kwargs,
        random_state=rng,
    )
    ica.fit(Xs)
    if n_components is None:
        n_components = min(n_features)
    assert ica.mixing_.shape == (n_components, n_components)
    assert ica.components_.shape == (n_components, n_components)
    for i, n_feature in enumerate(n_features):
        assert ica.means_[i].shape == (n_feature,)
        assert ica.individual_components_[i].shape == (n_components, n_feature)
        assert ica.individual_mixing_[i].shape == (n_feature, n_components)


@pytest.mark.parametrize("n_individual_components", ["auto", (2, 3, 2)])
@pytest.mark.parametrize("multiview_output", [True, False])
@pytest.mark.parametrize("solver", ["picard", "fastica"])
def test_source_recovery(n_individual_components, multiview_output, solver):
    rng = np.random.RandomState(0)
    n_samples = 500
    n_sources = 2
    n_features = [2, 3, 4]
    noise_level = 0.01
    Xs, sources, mixings = generate_signals(
        n_samples, n_sources, n_features, noise_level, rng
    )
    ica = GroupICA(
        n_components=2,
        n_individual_components=n_individual_components,
        multiview_output=multiview_output,
        solver=solver,
        random_state=rng,
    )
    ica.fit(Xs)
    estimated_sources = ica.transform(Xs)
    estimated_mixings = ica.individual_mixing_
    if multiview_output:
        for s in estimated_sources:
            C = np.dot(s.T, sources)
            assert amari_d(C) < 1e-3
        for A, A_ in zip(mixings, estimated_mixings):
            assert amari_d(np.linalg.pinv(A), A_) < 1e-3


@pytest.mark.parametrize("n_individual_components", ["auto", (2, 3, 2)])
@pytest.mark.parametrize("multiview_output", [True, False])
@pytest.mark.parametrize("solver", ["picard", "fastica"])
def test_inverse_transform(n_individual_components, multiview_output, solver):
    rng = np.random.RandomState(0)
    n_samples = 500
    n_sources = 2
    n_features = [2, 3, 4]
    noise_level = 0.0001
    Xs, sources, mixings = generate_signals(
        n_samples, n_sources, n_features, noise_level, rng
    )
    ica = GroupICA(
        n_components=2,
        n_individual_components=n_individual_components,
        multiview_output=multiview_output,
        solver=solver,
        random_state=rng,
    )
    ica.fit(Xs)
    estimated_sources = ica.transform(Xs)
    recovered_signals = ica.inverse_transform(estimated_sources)
    for X, X_ in zip(recovered_signals, Xs):
        assert_allclose(X, X_, atol=1e-2)
