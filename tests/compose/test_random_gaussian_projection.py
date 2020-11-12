"""
test_random_gaussian_projection.py
====================================
Tests functions in random_gaussian_projection.py.
"""

import numpy as np
from mvlearn.compose import RandomGaussianProjection

# IMPORTANT: Because random gaussian projection wraps sklearn's
# implementation, we do not test projection functionality. Instead
# we just focus on our parameter additions.


def test_multiple_views():
    n_views = 5
    X = np.random.rand(4, 4)
    rgp = RandomGaussianProjection(n_views=n_views, n_components=1)
    Xs = rgp.fit_transform(X)
    assert(len(Xs) == n_views)
    assert(len(rgp.GaussianRandomProjections_) == n_views)
    assert np.all([X.shape == (4, 1) for X in Xs])
