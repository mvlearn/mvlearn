import pytest
from multiview.cluster.base_cluster import BaseCluster


def test_base_cluster():
    base_cluster = BaseCluster()

    base_cluster.fit(Xs=None, y=None)
    base_cluster.predict(Xs=None)
    base_cluster.fit_predict(Xs=None, y=None)
