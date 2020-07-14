import pytest
from mvlearn.cluster.base import BaseCluster

def test_basecluster():
    bc = BaseCluster()
    bc.fit(Xs=None).predict(Xs=None)
