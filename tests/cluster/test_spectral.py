import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.neighbors import NearestNeighbors
from mvlearn.cluster import MultiviewSpectralClustering
from sklearn.exceptions import NotFittedError

# EXCEPTION TESTING

RANDOM_STATE=10
np.random.RandomState(RANDOM_STATE)


@pytest.fixture(scope='module')
def small_data():
    view1 = np.random.random((5, 8))
    view2 = np.random.random((5, 9))
    data = [view1, view2]
    return data

def test_n_clusters_not_positive_int(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=-1)
        spectral.fit_predict(small_data)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=0)
        spectral.fit_predict(small_data)

def test_random_state_not_convertible(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=5, random_state='ab')
        spectral.fit_predict(small_data)

def test_info_view_not_valid(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=2, info_view=-1)
        spectral.fit_predict(small_data)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=2, info_view=6)
        spectral.fit_predict(small_data)

def test_n_views_too_small1(small_data):
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1])

def test_n_views_too_small2(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([])
    
def test_samples_not_same(small_data):
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8))
        view2 = np.random.random((8, 9))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_samples_not_list(small_data):
    with pytest.raises(ValueError):
        view1 = 1
        view2 = 3
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_samples_not_2D_1(small_data):
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8, 7))
        view2 = np.random.random((5, 9, 7))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])
        
def test_samples_not_2D_2(small_data):
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_max_iter_not_positive_int(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(max_iter=-1)
        spectral.fit_predict(small_data)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(max_iter=0)
        spectral.fit_predict(small_data)
        
def test_n_init_not_positive_int(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_init=-1)
        spectral.fit_predict(small_data)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_init=0)
        spectral.fit_predict(small_data)
        
def test_not_valid_affinity(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='What')
        spectral.fit_predict(small_data)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity=None)
        spectral.fit_predict(small_data)
        
def test_gamma_not_positive_float(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(gamma=-1.5)
        spectral.fit_predict(small_data)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(gamma=0)
        spectral.fit_predict(small_data)
        
def test_n_neighbors_not_positive_int(small_data):
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='nearest_neighbors', n_neighbors=-1)
        spectral.fit_predict(small_data)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='nearest_neighbors', n_neighbors=0) 
        spectral.fit_predict(small_data)
        
# Function Testing
@pytest.fixture(scope='module')
def data():

    num_fit_samples = 200
    n_feats1 = 20
    n_feats2 = 18
    n_feats3 = 30
    n_clusters = 2
    np.random.seed(RANDOM_STATE)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))
    fit_data.append(np.random.rand(num_fit_samples, n_feats3))

    spectral = MultiviewSpectralClustering(n_clusters, random_state=RANDOM_STATE)
    return {'n_fit' : num_fit_samples, 'n_feats1': n_feats1, 'n_feats2': n_feats2,
            'n_feats3' : n_feats3, 'n_clusters': n_clusters, 'spectral' : spectral,
            'fit_data' : fit_data}

def test_affinity_mat_rbf(data):
        
    v1_data = data['fit_data'][0]
    spectral = data['spectral']

    distances = cdist(v1_data, v1_data)
    gamma = 1 / (2 * np.median(distances) ** 2)
    true_kernel = rbf_kernel(v1_data, gamma=gamma)
    g_kernel = spectral._affinity_mat(v1_data)

    assert(g_kernel.shape[0] == data['n_fit'])
    assert(g_kernel.shape[1] == data['n_fit'])

    for ind1 in range(g_kernel.shape[0]):
        for ind2 in range(g_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - g_kernel[ind1][ind2]) < 0.000001


def test_affinity_mat_rbf2(data):

    v1_data = data['fit_data'][0]
    gamma = 1
    spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE,
                                           gamma=gamma)
    distances = cdist(v1_data, v1_data)
    gamma = 1 / (2 * np.median(distances) ** 2)
    true_kernel = rbf_kernel(v1_data, gamma=1)
    g_kernel = spectral._affinity_mat(v1_data)

    assert(g_kernel.shape[0] == data['n_fit'])
    assert(g_kernel.shape[1] == data['n_fit'])

    for ind1 in range(g_kernel.shape[0]):
        for ind2 in range(g_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - g_kernel[ind1][ind2]) < 0.000001
            
def test_affinity_mat_poly(data):

    v1_data = data['fit_data'][0]

    distances = cdist(v1_data, v1_data)
    gamma = 1 / (2 * np.median(distances) ** 2)
    true_kernel = polynomial_kernel(v1_data, gamma=gamma)
    spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE,
                                           affinity='poly')
    p_kernel = spectral._affinity_mat(v1_data)

    assert(p_kernel.shape[0] == data['n_fit'])
    assert(p_kernel.shape[1] == data['n_fit'])

    for ind1 in range(p_kernel.shape[0]):
        for ind2 in range(p_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - p_kernel[ind1][ind2]) < 0.000001

def test_affinity_neighbors(data):

    v1_data = data['fit_data'][0]
    n_neighbors=10
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(v1_data)
    true_kernel = neighbors.kneighbors_graph(v1_data).toarray()
    spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE,
                        affinity='nearest_neighbors', n_neighbors=10)
    n_kernel = spectral._affinity_mat(v1_data)
    assert(n_kernel.shape[0] == data['n_fit'])
    assert(n_kernel.shape[1] == data['n_fit'])

    for ind1 in range(n_kernel.shape[0]):
        for ind2 in range(n_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - n_kernel[ind1][ind2]) < 0.000001

def test_compute_eigs(data):

    v1_data = data['fit_data'][0]
    g_kernel = rbf_kernel(v1_data, v1_data)
    n_clusts = data['n_clusters']
    n_fit = data['n_fit']

    spectral = data['spectral']
    eigs = spectral._compute_eigs(g_kernel)

    assert(eigs.shape[0] == n_fit)
    assert(eigs.shape[1] == n_clusts)

    col_mags = np.linalg.norm(eigs, axis=0)

    for val in col_mags:
        assert(np.abs(val - 1) < 0.000001)


def test_fit_predict_default(data):


    v_data = data['fit_data'][:2]
    spectral = MultiviewSpectralClustering(2, random_state=RANDOM_STATE)
    predictions = spectral.fit_predict(v_data)
    n_clusts = data['n_clusters']

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)

def test_fit_predict_n_views(data):        

    v_data = data['fit_data']
    spectral = data['spectral']
    predictions = spectral.fit_predict(v_data)
    n_clusts = data['n_clusters']

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)


def test_fit_predict_max_iter(data):


    v_data = data['fit_data']
    max_iter = 5
    n_clusts = data['n_clusters']
    spectral = MultiviewSpectralClustering(n_clusts,
                random_state=RANDOM_STATE, max_iter=max_iter)
    predictions = spectral.fit_predict(v_data)

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)


def test_fit_predict_info_view(data):

    v_data = data['fit_data']
    info_view = np.random.randint(len(v_data))
    n_clusts = data['n_clusters']
    spectral = MultiviewSpectralClustering(n_clusts,
                random_state=RANDOM_STATE, info_view=info_view)
    predictions = spectral.fit_predict(v_data)

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)


test_n_clusters_not_positive_int(small_data)