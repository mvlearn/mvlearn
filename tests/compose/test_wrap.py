import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from mvlearn.compose import ViewClassifier, ViewTransformer
import pytest


# Test ViewClassifier
def test_single_classifier():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 15
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    y = rng.choice(2, n_samples)
    clf = LogisticRegression(penalty='none')
    clfs = ViewClassifier(clf)
    clfs.fit(Xs, y)
    assert clfs.n_views_ == n_views
    assert len(clfs.estimators_) == n_views
    y_hat = clfs.predict(Xs)
    assert len(y_hat) == len(y)
    assert y_hat.ndim == 1
    assert_equal(clfs.score(Xs, y), 1)


def test_multiple_classifiers():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 15
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    y = rng.choice(2, n_samples)
    estimators = [LogisticRegression(penalty='none') for _ in range(n_views)]
    clfs = ViewClassifier(estimators)
    clfs.fit(Xs, y)
    assert clfs.n_views_ == n_views
    assert len(clfs.estimators_) == n_views
    y_hat = clfs.predict(Xs)
    assert len(y_hat) == len(y)
    assert y_hat.ndim == 1
    assert_equal(clfs.score(Xs, y), 1)


def test_error_classifiers():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    y = rng.choice(2, n_samples)
    estimators = [LogisticRegression() for _ in range(n_views-1)]
    clfs = ViewClassifier(estimators)
    with pytest.raises(ValueError):
        clfs.fit(Xs, y)


# Test ViewTransformer
def test_single_transformer():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    pca = PCA(n_components=2)
    repeat = ViewTransformer(pca)
    repeat.fit(Xs)
    assert repeat.n_views_ == n_views
    assert len(repeat.estimators_) == n_views
    assert (
        repeat.estimators_[0].components_[0, 0]
        != repeat.estimators_[1].components_[0, 0]
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
    estimators = [PCA(n_component) for n_component in n_components]
    repeat = ViewTransformer(estimators)
    repeat.fit(Xs)
    assert repeat.n_views_ == n_views
    assert len(repeat.estimators_) == n_views
    assert (
        repeat.estimators_[0].components_[0, 0]
        != repeat.estimators_[1].components_[0, 0]
    )
    X_transformed = repeat.transform(Xs)
    assert len(X_transformed) == n_views
    for X, n_component in zip(X_transformed, n_components):
        assert X.shape == (n_samples, n_component)
    X_orig = repeat.inverse_transform(X_transformed)
    for X, X2 in zip(Xs, X_orig):
        assert X.shape == X2.shape


def test_error_transformers():
    rng = np.random.RandomState(0)
    n_views = 4
    n_features = 5
    n_samples = 10
    Xs = rng.randn(n_views, n_samples, n_features)
    n_components = [2, 2]
    estimators = [PCA(n_component) for n_component in n_components]
    repeat = ViewTransformer(estimators)
    with assert_raises(ValueError):
        repeat.fit(Xs)
