import pytest
from mvlearn.cluster.base_kmeans import BaseKMeans


def test_base_kmeans():
    base_kmeans = BaseKMeans()

    base_kmeans.fit(Xs=None)
    base_kmeans.predict(Xs=None)
    base_kmeans.fit_predict(Xs=None)
