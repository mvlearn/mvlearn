import itertools
import warnings
import pytest

import numpy as np
from scipy import stats

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_warns

from sklearn.decomposition import FastICA, fastica, PCA
from sklearn.decomposition._fastica import _gs_decorrelation

from mvlearn.decomposition import GroupICA
from sklearn.exceptions import ConvergenceWarning


def amari_d(W, A):
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)

    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


@pytest.mark.parametrize("add_noise", [True, False])
@pytest.mark.parametrize("seed", range(1))
def test_group_ica_simple(add_noise, seed):
    rng = np.random.RandomState(seed)
    n_samples = 1000
    n_features = [4, 5, 6]
    n_sources = 3
    n_subjects = len(n_features)
    sources = rng.laplace(size=(n_samples, n_sources))
    mixings = [rng.randn(n_feature, n_sources) for n_feature in n_features]
    Xs = [np.dot(sources, mixing.T) for mixing in mixings]
    if add_noise:
        for i, (X, n_feature) in zip(Xs, n_featuress):
            Xs[i] = X + 0.01 * rng.randn(n_samples, n_feature)

    groupica = GroupICA(n_sources=n_sources).fit(Xs)
    estimated_sources = groupica.transform(Xs)
    mixings_ = groupica.mixings_
    assert estimated_sources.shape == (n_samples, n_sources)
    for i in range(n_subjects):
        assert mixings_[i].shape == (n_features[i], n_sources)

    # Check that sources are recovered
    corr = np.dot(estimated_sources.T, sources)
    distance = amari_d(corr, np.eye(n_sources))
    if add_noise:
        assert distance < .1
    else:
        assert distance < .05

    for i in range(n_subjects):
        distance = amari_d(np.linalg.pinv(mixings_[i]), mixings[i])
        if add_noise:
            assert distance < .1
        else:
            assert distance < .05
