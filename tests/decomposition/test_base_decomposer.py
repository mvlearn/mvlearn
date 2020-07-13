import sys
import pytest
from mvlearn.decomposition.base import BaseDecomposer


def test_base_decomposer():
    base = BaseDecomposer()
    base.fit(Xs=None, y=None).transform(Xs=None)
    base.fit_transform(Xs=None)
