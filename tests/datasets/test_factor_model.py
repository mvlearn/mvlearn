import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from mvlearn.datasets import sample_joint_factor_model


def test_shape():
    n_views = 5
    n_samples = 10
    n_features = 3
    Xs = sample_joint_factor_model(n_views, n_samples, n_features)
    assert len(Xs) == n_views
    assert np.all([X.shape[0] == n_samples for X in Xs])
    assert np.all([X.shape[1] == n_features for X in Xs])

    n_features = [1, 2, 3, 4, 5]
    Xs = sample_joint_factor_model(n_views, n_samples, n_features)
    assert np.all([X.shape[1] == f for X, f in zip(Xs, n_features)])


def test_result():
    n_views = 3
    n_samples = 10
    n_features = 5
    joint_rank = 3
    random_state = 0
    noise_std = 1
    m = 100
    Xs, U, loadings = sample_joint_factor_model(
        n_views, n_samples, n_features, joint_rank, return_decomp=True,
        random_state=random_state, m=m, noise_std=noise_std)
    assert_almost_equal(U.T @ U, np.eye(joint_rank))
    for loads in loadings:
        assert_almost_equal(loads.T @ loads, np.eye(joint_rank))

    Xs_hat = [X @ load for X, load in zip(Xs, loadings)]
    Xs_hat = [X / np.linalg.norm(X, axis=0) for X in Xs_hat]

    for X_hat in Xs_hat:
        assert_almost_equal(U, X_hat, decimal=2)
