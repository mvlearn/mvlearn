import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.neighbors import NearestNeighbors
from mvlearn.cluster.mv_spectral import MultiviewSpectralClustering
from sklearn.exceptions import NotFittedError

# EXCEPTION TESTING

RANDOM_STATE=10
np.random.RandomState(RANDOM_STATE)

def test_n_clusters_not_positive_int():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=-1)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=0)

def test_n_views_not_positive_int():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=5, n_views=-1)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=5, n_views=0)

def test_random_state_not_convertible():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=5, random_state='ab')


def test_info_view_not_valid():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=2, n_views=5, info_view=-1)
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_clusters=2, n_views=5, info_view=6)

def test_samples_not_same():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8))
        view2 = np.random.random((8, 9))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_samples_not_list():
    with pytest.raises(ValueError):
        view1 = 1
        view2 = 3
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_samples_not_2D_1():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8, 7))
        view2 = np.random.random((5, 9, 7))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])
        
def test_samples_not_2D_2():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        spectral = MultiviewSpectralClustering(random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])
        
def test_samples_not_n_views():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        spectral = MultiviewSpectralClustering(n_views=3, random_state=RANDOM_STATE)
        spectral.fit_predict([view1, view2])

def test_max_iter_not_positive_int():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(max_iter=-1)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(max_iter=0)

def test_n_init_not_positive_int():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_init=-1)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(n_init=0)

def test_not_valid_affinity():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='What')
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity=None)

def test_gamma_not_positive_float():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(gamma=-1.5)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(gamma=0)
        
def test_n_neighbors_not_positive_int():
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='nearest_neighbors', n_neighbors=-1)
        
    with pytest.raises(ValueError):
        spectral = MultiviewSpectralClustering(affinity='nearest_neighbors', n_neighbors=0) 
        
# Function Testing
@pytest.fixture(scope='module')
def data():

    num_fit_samples = 200
    n_feats1 = 20
    n_feats2 = 18
    n_feats3 = 30
    n_clusters = 2
    n_views = 3
    np.random.seed(RANDOM_STATE)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))
    fit_data.append(np.random.rand(num_fit_samples, n_feats3))

    spectral = MultiviewSpectralClustering(n_clusters, n_views=n_views, random_state=RANDOM_STATE)
    return {'n_fit' : num_fit_samples, 'n_feats1': n_feats1, 'n_feats2': n_feats2,
            'n_feats3' : n_feats3, 'n_clusters': n_clusters, 'spectral' : spectral,
            'fit_data' : fit_data, 'n_views' : n_views}

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
    n_views = data['n_views']
    gamma = 1
    spectral = MultiviewSpectralClustering(n_views=n_views,
                      random_state=RANDOM_STATE, gamma=gamma)
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
    n_views = data['n_views']

    distances = cdist(v1_data, v1_data)
    gamma = 1 / (2 * np.median(distances) ** 2)
    true_kernel = polynomial_kernel(v1_data, gamma=gamma)
    spectral = MultiviewSpectralClustering(n_views=n_views,
    random_state=RANDOM_STATE, affinity='poly')
    p_kernel = spectral._affinity_mat(v1_data)

    assert(p_kernel.shape[0] == data['n_fit'])
    assert(p_kernel.shape[1] == data['n_fit'])

    for ind1 in range(p_kernel.shape[0]):
        for ind2 in range(p_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - p_kernel[ind1][ind2]) < 0.000001

def test_affinity_neighbors(data):

    v1_data = data['fit_data'][0]
    n_views = data['n_views']
    n_neighbors=10
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(v1_data)
    true_kernel = neighbors.kneighbors_graph(v1_data).toarray()
    spectral = MultiviewSpectralClustering(n_views=n_views,
    random_state=RANDOM_STATE, affinity='nearest_neighbors', n_neighbors=10)
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
    n_views = data['n_views']
    max_iter = 5
    n_clusts = data['n_clusters']
    spectral = MultiviewSpectralClustering(n_clusts, n_views=n_views,
    random_state=RANDOM_STATE, max_iter=max_iter)
    predictions = spectral.fit_predict(v_data)

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)


def test_fit_predict_info_view(data):

    v_data = data['fit_data']
    n_views = data['n_views']
    info_view = np.random.randint(n_views)
    n_clusts = data['n_clusters']
    spectral = MultiviewSpectralClustering(n_clusts, n_views=n_views,
    random_state=RANDOM_STATE, info_view=info_view)
    predictions = spectral.fit_predict(v_data)

    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)

