import pytest
import numpy as np
from multiview.cluster.mv_k_means import MultiviewKMeans


# EXCEPTION TESTING

def test_k_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(k=-1)
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(k=0)
        
def test_random_state_not_convertible():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(random_state='ab')

def test_samples_not_same():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8))
        view2 = np.random.random((8, 9))
        kmeans = MultiviewKMeans()
        kmeans.fit([view1, view2])

def test_samples_not_2D_1():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8, 7))
        view2 = np.random.random((5, 9, 7))
        kmeans = MultiviewKMeans()
        kmeans.fit([view1, view2])

def test_samples_not_2D_2():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        kmeans = MultiviewKMeans()
        kmeans.fit([view1, view2])

def test_not_2_views():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        view3 = np.random.random((10,))
        kmeans = MultiviewKMeans()
        kmeans.fit([view1, view2, view3])

def test_patience_not_nonnegative_int():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans = MultiviewKMeans()
        kmeans.fit([view1, view2], patience=-1)

def test_max_iter_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans()
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans.fit([view1, view2], max_iter=-1)
    
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans()
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans.fit([view1, view2], max_iter=0)
        
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

    kmeans = MultiviewKMeans(random_state=random_seed)
    return {'n_test' : num_test_samples, 'kmeans' : kmeans,
            'fit_data' : fit_data, 'test_data' : test_data}


def test_predict(data):

    data['kmeans'].fit(data['fit_data'])
    cluster_pred = data['kmeans'].predict(data['test_data'])

    true_clusters = [4, 0, 0, 2, 2]

    for ind in range(data['n_test']):
        assert cluster_pred[ind] == true_clusters[ind]


def test_predict_patience(data):

    data['kmeans'].fit(data['fit_data'], patience=10)
    cluster_pred = data['kmeans'].predict(data['test_data'])

    true_clusters = [0, 2, 3, 0, 1]
    for ind in range(data['n_test']):
        assert cluster_pred[ind] == true_clusters[ind]


def test_predict_max_iter(data):

    data['kmeans'].fit(data['fit_data'], max_iter=4)
    cluster_pred = data['kmeans'].predict(data['test_data'])
    print(cluster_pred)
    true_clusters = [4, 4, 1, 4, 4]

    for ind in range(data['n_test']):
        assert cluster_pred[ind] == true_clusters[ind]
        

def test_compute_distance():

    n_samples = 5
    n_features = 3
    n_centroids = 2

    data = np.random.random((n_samples, n_features))
    centroids = np.random.random((n_centroids, n_features))
    kmeans = MultiviewKMeans(k=n_centroids)
    distances = kmeans._compute_distance(data, centroids)
    true_distances = [[0.94299374, 0.68062463, 0.48518824, 1.1816393,  1.26301881],
                       [0.50728258, 1.02524165, 0.39185344, 0.87996467, 0.92955915]]
    
    for ind1 in range(n_centroids):
        for ind2 in range(n_samples):
            assert np.abs(true_distances[ind1][ind2]
                          - distances[ind1][ind2]) < 0.000001
