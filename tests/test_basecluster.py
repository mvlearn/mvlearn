import sys
import pytest
from mvlearn.cluster.base import BaseCluster


def test_base_cluster():
    base = BaseCluster()
    labels = base.fit_predict(Xs=None, y=None)
