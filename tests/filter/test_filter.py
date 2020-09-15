import numpy as np
from numpy.testing import assert_raises

import pytest

from mvlearn.filter import Filter
from sklearn.decomposition import PCA


def test_single_transformer():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    pca = PCA(n_components=2)
    filt = Filter(pca)
    filt.fit(Xs)
    assert filt.n_views_ == n_views
    assert len(filt.transformer_list_) == n_views
    assert (
        filt.transformer_list_[0].components_[0, 0]
        != filt.transformer_list_[1].components_[0, 0]
    )
    X_transformed = filt.transform(Xs)
    assert len(X_transformed) == n_views
    for X in X_transformed:
        assert X.shape == (n_samples, 2)
    X_orig = filt.inverse_transform(X_transformed)

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
    filt = Filter(transformers)
    filt.fit(Xs)
    assert filt.n_views_ == n_views
    assert len(filt.transformer_list_) == n_views
    assert (
        filt.transformer_list_[0].components_[0, 0]
        != filt.transformer_list_[1].components_[0, 0]
    )
    X_transformed = filt.transform(Xs)
    assert len(X_transformed) == n_views
    for X, n_component in zip(X_transformed, n_components):
        assert X.shape == (n_samples, n_component)
    X_orig = filt.inverse_transform(X_transformed)
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
    filt = Filter(transformers)
    with assert_raises(ValueError):
        filt.fit(Xs)
