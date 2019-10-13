import pytest
import numpy as np
from multiview.cluster.k_means import KMeans


# EXCEPTION TESTING

def test_k_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = KMeans(k=-1)
    with pytest.raises(ValueError):
        kmeans = KMeans(k=0)

def test_max_iter_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = KMeans(max_iter=-1)
    with pytest.raises(ValueError):
        kmeans = KMeans(max_iter=0)
        
def test_random_state_not_convertible():
    with pytest.raises(ValueError):
        kmeans = KMeans(random_state='ab')

def test_samples_not_same():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8))
        view2 = np.random.random((8, 9))
        kmeans = KMeans()
        kmeans.fit([view1, view2])

def test_samples_not_2D_1():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8, 7))
        view2 = np.random.random((5, 9, 7))
        kmeans = KMeans()
        kmeans.fit([view1, view2])

def test_samples_not_2D_2():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        kmeans = KMeans()
        kmeans.fit([view1, view2])

def test_not_2_views():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        view3 = np.random.random((10,))
        kmeans = KMeans()
        kmeans.fit([view1, view2, view3])

def test_patience_not_nonnegative_int():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans = KMeans()
        kmeans.fit([view1, view2], patience=-1)

# Function Testing

@pytest.fixture(scope='module')
def data():
    random_seed = 1
    num_fit_samples = 200
    num_test_samples = 5
    n_feats1 = 20
    n_feats2 = 18
    np.random.seed(random_seed)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))

    test_data = []
    test_data.append(np.random.rand(num_test_samples, n_feats1))
    test_data.append(np.random.rand(num_test_samples, n_feats2))

    kmeans = KMeans(random_state=random_seed)
    return {'n_test' : num_test_samples, 'kmeans' : kmeans,
            'fit_data' : fit_data, 'test_data' : test_data}


def test_predict(data):

    data['kmeans'].fit(data['fit_data'])
    cluster_pred = data['kmeans'].predict(data['test_data'])

    true_clusters = [4, 0, 0, 2, 2]

    for ind in range(data['n_test']):
        assert cluster_pred[ind] == true_clusters[ind]


def test_compute_distance():

    n_samples = 5
    n_features = 3
    n_centroids = 2

    data = np.random.random((n_samples, n_features))
    centroids = np.random.random((n_centroids, n_features))
    kmeans = KMeans(k=n_centroids)
    distances = kmeans._compute_distance(data, centroids)
    true_distances = [[1.056409, 0.607753, 0.803512, 1.072, 0.889953],
                       [0.929535, 0.449823, 0.680247, 0.896337, 0.755213]]

    for ind1 in range(n_centroids):
        for ind2 in range(n_samples):
            assert np.abs(true_distances[ind1][ind2]
                          - distances[ind1][ind2]) < 0.000001
