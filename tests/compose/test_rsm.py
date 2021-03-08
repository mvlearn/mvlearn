"""
test_rsm.py
====================================
Tests Random subspace method for view construction and
param check helper functions.
"""

import numpy as np
from mvlearn.compose import RandomSubspaceMethod
import pytest


def test_rsm_return_vals():
    X = np.random.rand(25, 25)
    n_views = 4
    rsm = RandomSubspaceMethod(n_views, 10)
    Xs = rsm.fit_transform(X)

    # test that 4 Xs are returned
    assert len(Xs) == n_views
    assert len(rsm.subspace_indices_) == n_views
    assert rsm.subspace_dim_ == 10
    # test that each view has correct shape
    assert np.all([X.shape == (25, 10) for X in Xs])


def test_rsm_with_float_n_features():
    X = np.random.rand(4, 4)
    n_views = 4
    rsm = RandomSubspaceMethod(n_views, 0.5)
    Xs = rsm.fit_transform(X)

    # test that 4 Xs were returned
    assert len(Xs) == n_views
    assert len(rsm.subspace_indices_) == n_views
    assert rsm.subspace_dim_ == int(0.5 * n_views)
    # test that each view has correct shape
    assert np.all([X.shape == (n_views, int(0.5 * n_views)) for X in Xs])


def test_features_fail():
    X = np.random.rand(25, 25)
    n_views = 4
    rsm = RandomSubspaceMethod(n_views, 10)
    rsm = rsm.fit(X)
    with pytest.raises(ValueError):
        rsm.transform(X[:, :-2])
