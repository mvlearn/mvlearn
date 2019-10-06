import sys
sys.path.append("../")

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from multiview.utils.utils import check_Xs_y, check_Xs

def test_good_input():
    np.random.seed(1)
    test_Xs = np.array([[1,2],[3,4]])
    test_y = [1,2]

    def single_view_X():
        assert len(check_Xs(test_Xs, multiview=False)) == 1, ""

    def single_view_y():
        Xs,y = check_Xs_y(test_Xs, test_y, multiview=False)
        assert len(y) == 2, ""
        assert len(Xs) == 1, ""

    single_view_X()
    single_view_y()

def test_bad_inputs():
    np.random.seed(1)
    test_Xs = np.array([[1,2],[3,4]])
    test_y = [1,2]

    with pytest.raises(ValueError):
        "Test single graph input"
        check_Xs(test_Xs, multiview=True)

    with pytest.raises(ValueError):
        "Test empty input"
        check_Xs_y(test_Xs,test_y,multiview=True)