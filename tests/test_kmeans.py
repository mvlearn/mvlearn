import pytest
import numpy as np
from multiview.cluster.mv_k_means import MultiviewKMeans
from sklearn.exceptions import NotFittedError

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

def test_samples_not_list():
    with pytest.raises(ValueError):
        view1 = 1
        view2 = 3
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

def test_not_fit():
    with pytest.raises(NotFittedError):
        kmeans = MultiviewKMeans()
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans.predict([view1, view2])
        
# Function Testing

@pytest.fixture(scope='module')
def data_random():
    
    random_seed = 1
    num_fit_samples = 200
    num_test_samples = 5
    n_feats1 = 20
    n_feats2 = 18
    n_clusters = 3
    np.random.seed(random_seed)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))

    test_data = []
    test_data.append(np.random.rand(num_test_samples, n_feats1))
    test_data.append(np.random.rand(num_test_samples, n_feats2))

    kmeans = MultiviewKMeans(k=n_clusters, random_state=random_seed)
    return {'n_test' : num_test_samples, 'n_feats1': n_feats1, 'n_feats2': n_feats2,
            'n_clusters': n_clusters, 'kmeans' : kmeans, 'fit_data' : fit_data,
            'test_data' : test_data}
    
def test_compute_distance():

    n_centroids = 2
    n_samples = 3
    centroids = np.array([[5,3,-2], [8, 0, 1.5]])
    data = np.array([[5, 3, -2,], [0, 0, 0], [1, 11, -4]])
    kmeans = MultiviewKMeans(k=n_centroids)
    distances = kmeans._compute_distance(data, centroids)
    true_distances = np.array([[ 0, 6.164414, 9.16515139],
                       [ 5.5, 8.1394103, 14.1509717 ]])
    
    for ind1 in range(n_centroids):
        for ind2 in range(n_samples):
            assert np.abs(true_distances[ind1][ind2]
                          - distances[ind1][ind2]) < 0.000001


def test_compute_positive():

    n_centroids = 3
    n_samples = 10
    n_feats = 20

    samples = np.random.rand(n_samples, n_feats)
    centroids = np.random.rand(n_centroids, n_feats)
    kmeans = MultiviewKMeans(k=n_centroids)
    distances = kmeans._compute_distance(samples, centroids)

    for dist in distances.flatten():
        assert(dist >= 0)

            
def test_fit_centroids(data_random):
    kmeans = data_random['kmeans']
    kmeans.fit(data_random['fit_data'])

    assert(len(kmeans._centroids) == 2)
    assert(kmeans._centroids[0].shape[0] == data_random['n_clusters'])
    assert(kmeans._centroids[1].shape[0] == data_random['n_clusters'])
    
    for cent in kmeans._centroids[0]:
        assert(cent.shape[0] == data_random['n_feats1'])
    for cent in kmeans._centroids[1]:
        assert(cent.shape[0] == data_random['n_feats2'])


        
def test_predict_random(data_random):

    kmeans = data_random['kmeans']
    kmeans.fit(data_random['fit_data'])
    cluster_pred = kmeans.predict(data_random['test_data'])

    assert(data_random['n_test'] ==  cluster_pred.shape[0])

    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])


def test_predict_random_small(data_random):

    kmeans = data_random['kmeans']
    kmeans.fit(data_random['fit_data'][:2])
    cluster_pred = kmeans.predict(data_random['test_data'])

    assert(data_random['n_test'] ==  cluster_pred.shape[0])

    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])

def test_predict_deterministic():

    n_clusters = 2
    v1_centroid = np.array([[0, 0],[1, 1]])
    v2_centroid = np.array([[0, 0],[1, 1]])
    centroids = [v1_centroid, v2_centroid]
    v1_data = np.array([[0, 0],[0.3, 0.2],[0.5, 0.5],[0.7, 0.7],[1, 1]])
    v2_data = np.array([[0, 0],[0.2, 0.4],[0.5, 0.5],[0.4, 0.7],[1, 1]])
    data = [v1_data, v2_data]
    kmeans = MultiviewKMeans(k=n_clusters)
    kmeans._centroids = centroids
    cluster_pred = kmeans.predict(data)
    true_clusters = [0, 0, 0, 1, 1]

    for ind in range(len(true_clusters)):
        assert cluster_pred[ind] == true_clusters[ind]



def test_fit_predict(data_random):
    
    kmeans = data_random['kmeans']
    cluster_pred = kmeans.fit_predict(data_random['test_data'])
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])
    
def test_fit_predict_patience(data_random):

    kmeans = data_random['kmeans']
    cluster_pred = kmeans.fit_predict(data_random['test_data'], patience=10)
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])

def test_fit_predict_max_iter(data_random):

    kmeans = data_random['kmeans']
    cluster_pred = kmeans.fit_predict(data_random['test_data'], max_iter=4)
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])

