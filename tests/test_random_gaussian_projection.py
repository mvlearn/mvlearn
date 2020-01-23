"""
test_random_gaussian_projection.py
====================================
Tests functions in random_gaussian_projection.py.
"""

import pytest
import numpy as np
from mvlearn.construct import random_gaussian_projection

# IMPORTANT: Because random gaussian projection wraps sklearn's
# implementation, we do not test projection functionality. Instead
# we just focus on our parameter additions.

def test_multiple_views():
    n_views = 5
    X = np.random.rand(4, 4)
    views = random_gaussian_projection(X, n_views=n_views, n_components=1)
    assert(len(views) == n_views)
