import sys
sys.path.append("../..")

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from multiview.embed.gcca import GCCA

def test_output():
    def _get_Xs(n_views = 2):
        np.random.seed(0)
        n_obs = 4
        n_features = 6
        X = np.random.normal(0, 1, size=(n_views, n_obs, n_features))
        return(X)

    def _compute_dissimilarity(arr):
        n = len(arr)
        out = np.zeros((n, n))
        for i in range(n):
            out[i] = np.linalg.norm(arr - arr[i])

        return out

    def use_fit_transform():
        n = 2
        Xs = _get_Xs(n)

        projs = GCCA().fit_transform(Xs)

        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit():
        n = 2
        Xs = _get_Xs(n)
        
        gcca = GCCA().fit(Xs)
        projs = gcca.fit_transform(Xs)

        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    use_fit_transform()
    use_fit()

def test_bad_inputs():
    np.random.seed(1)
    test_mat = np.array([[1,2],[3,4]])

    with pytest.raises(ValueError):
        "Test single graph input"
        GCCA().fit(test_mat)

    with pytest.raises(ValueError):
        "Test empty input"
        GCCA().fit(np.array([[]]))