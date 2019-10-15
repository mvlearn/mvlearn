import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from multiview.embed.gcca import GCCA


def test_output():
    def _get_Xs(n_views=2):
        np.random.seed(0)
        n_obs = 4
        n_features = 6
        X = np.random.normal(0, 1, size=(n_views, n_obs, n_features))
        return X

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


test_mat = np.array([[1, 2], [3, 4]])
mat_good = np.ones((2, 4, 2))


@pytest.mark.parameterize(
    "params,err",
    [
        ({"Xs": [[]]}, ValueError),  # Empty input
        ({"Xs": test_mat}, ValueError),  # Single matrix input
        ({"Xs": mat_good, "fraction_var": "fail"}, TypeError),
        ({"Xs": mat_good, "fraction_var": -1}, ValueError),
        ({"Xs": mat_good, "n_components": "fail"}, TypeError),
        ({"Xs": mat_good, "n_components": -1}, ValueError),
        ({"Xs": mat_good, "sv_tolerance": "fail"}, TypeError),
        ({"Xs": mat_good, "sv_tolerance": -1}, ValueError),
        ({"Xs": mat_good, "n_components": mat_good.shape[1]}, ValueError),
    ],
)
def test_bad_inputs(params, err):
    np.random.seed(1)
    with pytest.raises(err):
        GCCA().fit(**params)
