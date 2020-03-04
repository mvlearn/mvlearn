"""
test_rsm.py
====================================
Tests Random subspace method for view construction and 
param check helper functions.
"""

import pytest
import numpy as np
from mvlearn.construct import rsm

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
