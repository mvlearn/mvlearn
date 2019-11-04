import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from multiview.utils.utils import check_Xs_y, check_Xs, check_Xs_y_nan_allowed


def test_good_input():
    np.random.seed(1)
    test_Xs = np.array([[1, 2], [3, 4]])
    test_y = [1, 2]
    test_Xs2 = np.array([[1, 2], [3, 4], [5, 6]])
    test_y2 = [1, 2, np.nan]

    def single_view_X():
        assert_equal(len(check_Xs(test_Xs, multiview=False)), 1)
        assert_equal(len(check_Xs([test_Xs], multiview=False)), 1)

    def single_view_y():
        Xs, y = check_Xs_y(test_Xs, test_y, multiview=False)
        assert_equal(len(y), 2)
        assert_equal(len(Xs), 1)

    def single_view_y_nan_allowed():
        test_y_nan = test_y.copy()
        test_y_nan[0] = np.nan
        Xs, y = check_Xs_y_nan_allowed(test_Xs, test_y_nan, multiview=False)
        assert_equal(len(y), 2)
        assert_equal(len(Xs), 1)

    def multi_view_nan_allowed():
        Xs, y = check_Xs_y_nan_allowed(test_Xs2, test_y2, multiview=True,
                                       enforce_views=2, num_classes=2)

    single_view_X()
    single_view_y()
    single_view_y_nan_allowed()

def test_bad_inputs():
    np.random.seed(1)
    test_Xs = np.array([[1, 2], [3, 4]])
    test_y = [1, 2]
    bad_y = [1, 2, np.nan]
    bad_y2 = [1, 1, np.nan]
    test_Xs2 = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        "Test single graph input"
        check_Xs(test_Xs, multiview=True)

    with pytest.raises(ValueError):
        "Nonlist input"
        check_Xs('wrong', multiview=True)

    with pytest.raises(ValueError):
        "Test empty input"
        check_Xs([], multiview=True)

    with pytest.raises(ValueError):
        "Test different number of samples"
        check_Xs([test_Xs, [[1, 2]]], multiview=True)

    with pytest.raises(ValueError):
        "Test different number of samples on nan_allowed"
        check_Xs_y_nan_allowed([test_Xs, [[1, 2]]], test_y, multiview=True)

    with pytest.raises(ValueError):
        "Bad label length on nan_allowed"
        check_Xs_y_nan_allowed([test_Xs, [[1, 2], [4, 5]]], bad_y, multiview=True)

    with pytest.raises(ValueError):
        "Enforce wrong number of views"
        check_Xs([test_Xs, [[1, 2], [4, 5]]], multiview=True,
                 enforce_views=3)

    with pytest.raises(ValueError):
        "Enforce wrong number of views"
        check_Xs([test_Xs, [[1, 2], [4, 5]]], multiview=True,
                 enforce_views=3)

    with pytest.raises(ValueError):
        "Enforce wrong number of classes"
        check_Xs_y_nan_allowed([test_Xs2, [[1, 2], [4, 5]]], bad_y2, multiview=True,
                 enforce_views=2, num_classes=3)
