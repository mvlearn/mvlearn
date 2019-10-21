import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from multiview.embed.pls import PLS_embedding

np.random.seed(0)
X = [[1, 1, 1], [1, 2, 2], [1, 2, 2]]
Y = np.random.normal(10, 1, size=(3, 4))


def test_default():
    projs = PLS_embedding(X, Y, n_components=2, return_embedding=True)
    assert_equal(projs[:, 1], np.zeros(shape=projs[:, 1].shape))


def test_return_emedding():
    projs = PLS_embedding(X, Y, n_components=2, return_embedding=False)
    assert_equal(projs[:, 1], np.zeros(shape=projs[:, 1].shape))
