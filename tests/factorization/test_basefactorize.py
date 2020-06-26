import sys
import pytest
from mvlearn.factorization.base import BaseFactorize


def test_base_factorize():
    base = BaseFactorize()
    base.fit(Xs=None, y=None).transform(Xs=None)
    base.fit_transform(Xs=None)
