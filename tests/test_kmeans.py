import pytest
import numpy as np
from multiview.cluster.mv_k_means import MultiviewKMeans
from sklearn.exceptions import NotFittedError, ConvergenceWarning

# EXCEPTION TESTING
RANDOM_SEED = 10

def test_n_clusters_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(n_clusters=-1)
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(n_clusters=0)
        
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
        kmeans = MultiviewKMeans(patience=-1)

def test_max_iter_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(max_iter=-1)
    
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(max_iter=0)

        
def test_n_init_not_positive_int():
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(n_init=-1)
    
    with pytest.raises(ValueError):
        kmeans = MultiviewKMeans(n_init=0)

        
def test_final_centroids_no_consensus():
    with pytest.raises(ConvergenceWarning):
        kmeans = MultiviewKMeans(random_state=RANDOM_SEED)
        view1 = np.array([[0, 1], [1, 0]])
        view2 = np.array([[1, 0], [0, 1]])
        v1_centroids = np.array([[0, 1],[1, 0]])
        v2_centroids = np.array([[0, 1],[1, 0]])
        centroids = [v1_centroids, v2_centroids]
        kmeans._final_centroids([view1, view2], centroids)

def test_final_centroids_less_than_n_clusters():
    with pytest.raises(ConvergenceWarning):
        kmeans = MultiviewKMeans(n_clusters=3, random_state=RANDOM_SEED)
        view1 = np.random.random((2,5))
        view2 = np.random.random((2,6))
        v1_centroids = np.random.random((3, 5))
        v2_centroids = np.random.random((3, 6))
        centroids = [v1_centroids, v2_centroids]
        kmeans._final_centroids([view1, view2], centroids)

def test_final_centroids_less_than_n_clusters():
    with pytest.raises(ConvergenceWarning):
        kmeans = MultiviewKMeans(n_clusters=3, random_state=RANDOM_SEED)
        view1 = np.random.random((2,11))
        view2 = np.random.random((2,10))
        kmeans.fit([view1, view2])


def test_predict_not_fit():
    with pytest.raises(NotFittedError):
        kmeans = MultiviewKMeans()
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans.predict([view1, view2])

def test_predict_no_centroids1():
    with pytest.raises(AttributeError):
        kmeans = MultiviewKMeans()
        kmeans.centroids_ = [None, None]
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        kmeans.predict([view1, view2]) 

def test_predict_no_centroids2():
    kmeans = MultiviewKMeans()
    
    with pytest.raises(ConvergenceWarning):
        view1 = np.array([[0, 1], [1, 0]])
        view2 = np.array([[1, 0], [0, 1]])
        v1_centroids = np.array([[0, 1],[1, 0]])
        v2_centroids = np.array([[0, 1],[1, 0]])
        centroids = [v1_centroids, v2_centroids]
        kmeans._final_centroids([view1, view2], centroids)

    with pytest.raises(AttributeError):
        kmeans.predict([view1, view2])

# Function Testing

@pytest.fixture(scope='module')
def data_random():
    
    num_fit_samples = 200
    num_test_samples = 5
    n_feats1 = 20
    n_feats2 = 18
    n_clusters = 3
    np.random.seed(RANDOM_SEED)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))

    test_data = []
    test_data.append(np.random.rand(num_test_samples, n_feats1))
    test_data.append(np.random.rand(num_test_samples, n_feats2))

    kmeans = MultiviewKMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    return {'n_test' : num_test_samples, 'n_feats1': n_feats1, 'n_feats2': n_feats2,
            'n_clusters': n_clusters, 'kmeans' : kmeans, 'fit_data' : fit_data,
            'test_data' : test_data}
    
            
def test_fit_centroids(data_random):
    kmeans = data_random['kmeans']
    kmeans.fit(data_random['fit_data'])

    assert(len(kmeans.centroids_) == 2)
    assert(kmeans.centroids_[0].shape[0] == data_random['n_clusters'])
    assert(kmeans.centroids_[1].shape[0] == data_random['n_clusters'])
    
    for cent in kmeans.centroids_[0]:
        assert(cent.shape[0] == data_random['n_feats1'])
    for cent in kmeans.centroids_[1]:
        assert(cent.shape[0] == data_random['n_feats2'])

        
def test_predict_random(data_random):

    kmeans = data_random['kmeans']
    kmeans.fit(data_random['fit_data'])
    cluster_pred = kmeans.predict(data_random['test_data'])

    assert(data_random['n_test'] ==  cluster_pred.shape[0])

    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])


def test_predict_random_small(data_random):

    kmeans = MultiviewKMeans()
    input_data = [data_random['fit_data'][0][:2],data_random['fit_data'][1][:2]] 
    kmeans.fit(input_data)
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
    kmeans = MultiviewKMeans(n_clusters=n_clusters)
    kmeans.centroids_ = centroids
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

    
    n_clusters = data_random['n_clusters']
    patience=10
    kmeans = MultiviewKMeans(n_clusters=n_clusters, patience=patience)
    cluster_pred = kmeans.fit_predict(data_random['test_data'])
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])

def test_fit_predict_max_iter(data_random):

    
    n_clusters = data_random['n_clusters']
    max_iter = 5
    kmeans = MultiviewKMeans(n_clusters=n_clusters, max_iter=max_iter)
    cluster_pred = kmeans.fit_predict(data_random['test_data'])
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])


def test_fit_predict_n_init(data_random):

    
    n_clusters = data_random['n_clusters']
    n_init=1
    kmeans = MultiviewKMeans(n_clusters=n_clusters, n_init=n_init)
    cluster_pred = kmeans.fit_predict(data_random['test_data'])
    
    assert(data_random['n_test'] ==  cluster_pred.shape[0])
    for cl in cluster_pred:
        assert(cl >= 0 and cl < data_random['n_clusters'])
