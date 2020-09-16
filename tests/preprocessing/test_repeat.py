import numpy as np
from numpy.testing import assert_raises, assert_allclose

import pytest

from mvlearn.preprocessing import ViewTransformed
from sklearn.decomposition import PCA


def test_single_transformer():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    pca = PCA(n_components=2)
    repeat = ViewTransformed(pca)
    repeat.fit(Xs)
    assert repeat.n_views_ == n_views
    assert len(repeat.transformers_) == n_views
    assert (
        repeat.transformers_[0].components_[0, 0]
        != repeat.transformers_[1].components_[0, 0]
    )
    X_transformed = repeat.transform(Xs)
    assert len(X_transformed) == n_views
    for X in X_transformed:
        assert X.shape == (n_samples, 2)
    X_transformed2 = repeat.fit_transform(Xs)
    for X, X2 in zip(X_transformed, X_transformed2):
        assert_allclose(X, X2)
    X_orig = repeat.inverse_transform(X_transformed)

    for X, X2 in zip(Xs, X_orig):
        assert X.shape == X2.shape


def test_multiple_transformers():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    n_components = [2, 2, 3, 4]
    transformers = [PCA(n_component) for n_component in n_components]
    repeat = ViewTransformed(transformers)
    repeat.fit(Xs)
    assert repeat.n_views_ == n_views
    assert len(repeat.transformers_) == n_views
    assert (
        repeat.transformers_[0].components_[0, 0]
        != repeat.transformers_[1].components_[0, 0]
    )
    X_transformed = repeat.transform(Xs)
    assert len(X_transformed) == n_views
    for X, n_component in zip(X_transformed, n_components):
        assert X.shape == (n_samples, n_component)
    X_orig = repeat.inverse_transform(X_transformed)
    for X, X2 in zip(Xs, X_orig):
        assert X.shape == X2.shape


def test_error():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    n_components = [2, 2]
    transformers = [PCA(n_component) for n_component in n_components]
    repeat = ViewTransformed(transformers)
    with assert_raises(ValueError):
        repeat.fit(Xs)
