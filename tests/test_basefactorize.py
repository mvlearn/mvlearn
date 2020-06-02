import sys
import pytest
from mvlearn.factorization.base import BaseFactorize


def test_base_embed():
    base = BaseFactorize()
    base.fit(Xs=None, y=None)