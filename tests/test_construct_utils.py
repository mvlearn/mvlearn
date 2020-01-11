"""
test_construct_utils.py
====================================
Test mvlearn.construct.utils.py
"""

import pytest
from mvlearn.construct import utils


def test_check_n_views():
    # this should just pass
    utils.check_n_views(1)
    utils.check_n_views(2)

    # check that exception was raised in following
    # number of views is 0
    with pytest.raises(Exception):
        utils.check_n_views(0)

    # number of views is negative
    with pytest.raises(Exception):
        utils.check_n_views(-1)

    # number of views is not int
    with pytest.raises(Exception):
        utils.check_n_views(2.5)


def test_check_n_features():
    #these should all pass:
    cols = 10
    utils.check_n_features(1, cols)
    utils.check_n_features(.5, cols)

    # n_features > cols
    with pytest.raises(Exception):
        utils.check_n_features(11, cols)
    
    # check <= 0
    with pytest.raises(Exception):
        utils.check_n_features(0, cols)
    with pytest.raises(Exception):
        utils.check_n_features(-1, cols)

    # check int if > 1
    with pytest.raises(Exception):
        utils.check_n_features(1.5, cols)
