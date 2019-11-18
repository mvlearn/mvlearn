import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy.linalg import orth

from multiview.embed.gcca import GCCA


def generate_data(n=10, elbows=3, seed=1):
    """
    Generate data matrix with a specific number of elbows on scree plot
    """
    np.random.seed(seed)
    x = np.random.binomial(1, 0.6, (n ** 2)).reshape(n, n)
    xorth = orth(x)
    d = np.zeros(xorth.shape[0])
    for i in range(0, len(d), int(len(d) / (elbows + 1))):
        d[:i] += 10
    A = xorth.T.dot(np.diag(d)).dot(xorth)
    return A, d


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

        projs = GCCA().fit_transform(Xs, fraction_var=0.9)
        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit_view_idx():
        n = 2
        Xs = _get_Xs(n)

        gcca = GCCA().fit(Xs, fraction_var=0.9)
        projs = [gcca.transform(Xs[i], view_idx=i) for i in range(n)]

        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit_tall():
        n = 2
        Xs = _get_Xs(n)

        projs = GCCA().fit_transform(Xs, fraction_var=0.9, tall=True)
        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit_n_components():
        n = 2
        Xs = _get_Xs(n)

        projs = GCCA().fit_transform(Xs, n_components=3)
        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit_sv_tolerance():
        n = 2
        Xs = _get_Xs(n)

        projs = GCCA().fit_transform(Xs, sv_tolerance=1)
        dists = _compute_dissimilarity(projs)

        # Checks up to 7 decimal points
        assert_almost_equal(np.zeros((n, n)), dists)

    def use_fit_elbows():
        n = 2
        X, _ = generate_data(10, 3)
        Xs = [X, X]

        gcca = GCCA()
        projs = gcca.fit_transform(Xs, n_elbows=2)

        assert_equal(gcca._ranks[0], 4)

    use_fit_transform()
    use_fit_tall()
    use_fit_n_components()
    use_fit_sv_tolerance()
    use_fit_view_idx()
    use_fit_elbows()


test_mat = np.array([[1, 2], [3, 4]])
mat_good = np.ones((2, 4, 2))
Xs = np.random.normal(0, 1, size=(2, 4, 6))


@pytest.mark.parametrize(
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
    with pytest.raises(err):
        np.random.seed(1)
        GCCA().fit(**params)


def test_no_fit(params={"Xs": mat_good}, err=RuntimeError):
    with pytest.raises(err):
        np.random.seed(1)
        GCCA().transform(**params)
