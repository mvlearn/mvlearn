"""
test_rsm.py
====================================
Tests Random subspace method for view construction and 
param check helper functions.
"""

import pytest
import numpy as np
from multiview.construct import rsm

def test_rsm_return_vals():
    X = np.random.rand(25, 25)
    views = rsm.random_subspace_method(X, 10, 4)

    # test that 4 views are returned
    assert len(views) == 4

    # test that each view has correct shape
    for view in views:
        assert view.shape == (25, 25)
    
    # test that each view has correct number of cols set to 0
    for view in views:
        assert (np.where(~view.any(axis = 0))[0]).size == 15


def test_rsm_with_float_n_features():
    X = np.random.rand(4, 4)
    views = rsm.random_subspace_method(X, .5, 4)

    # test that 4 views were returned
    assert len(views) == 4

    # test that each view has correct number of cols set to 0
    for view in views:
        assert (np.where(~view.any(axis = 0))[0]).size == 2


def test_check_n_views():
    # this should just pass
    rsm.check_n_views(1)
    rsm.check_n_views(2)

    # check that exception was raised in following
    # number of views is 0
    with pytest.raises(Exception):
        rsm.check_n_views(0)

    # number of views is negative
    with pytest.raises(Exception):
        rsm.check_n_views(-1)

    # number of views is not int
    with pytest.raises(Exception):
        rsm.check_n_views(2.5)


def test_check_n_features():
    #these should all pass:
    cols = 10
    rsm.check_n_features(1, cols)
    rsm.check_n_features(.5, cols)

    # n_features > cols
    with pytest.raises(Exception):
        rsm.check_n_features(11, cols)
    
    # check <= 0
    with pytest.raises(Exception):
        rsm.check_n_features(0, cols)
    with pytest.raises(Exception):
        rsm.check_n_features(-1, cols)

    # check int if > 1
    with pytest.raises(Exception):
        rsm.check_n_features(1.5, cols)





