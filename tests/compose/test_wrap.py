import numpy as np
from numpy.testing import assert_equal
from mvlearn.compose import ViewClassifier
from sklearn.linear_model import LogisticRegression
import pytest


def test_single_estimator():
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


def test_multiple_estimators():
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


def test_error():
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
