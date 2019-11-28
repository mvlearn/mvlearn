import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from multiview.cluster.mv_spectral import MultiviewSpectralClustering
from sklearn.exceptions import NotFittedError

# EXCEPTION TESTING

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
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2])

def test_samples_not_list():
    with pytest.raises(ValueError):
        view1 = 1
        view2 = 3
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2])
        
def test_samples_not_2D_1():
    with pytest.raises(ValueError):
        view1 = np.random.random((5, 8, 7))
        view2 = np.random.random((5, 9, 7))
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2])

def test_samples_not_2D_2():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,))
        view2 = np.random.random((10,))
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2])


def test_samples_not_n_views():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        spectral = MultiviewSpectralClustering(2, n_views=3)
        spectral.fit_predict([view1, view2])
        

def test_n_iter_not_positive_int():
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2], n_iter=-1)

    
    with pytest.raises(ValueError):
        view1 = np.random.random((10,11))
        view2 = np.random.random((10,10))
        spectral = MultiviewSpectralClustering(2)
        spectral.fit_predict([view1, view2], n_iter=0)


        
# Function Testing



@pytest.fixture(scope='module')
def data():
    
    random_seed = 1
    num_fit_samples = 200
    n_feats1 = 20
    n_feats2 = 18
    n_feats3 = 30
    n_clusters = 2
    n_views = 3
    np.random.seed(random_seed)
    fit_data = []
    fit_data.append(np.random.rand(num_fit_samples, n_feats1))
    fit_data.append(np.random.rand(num_fit_samples, n_feats2))
    fit_data.append(np.random.rand(num_fit_samples, n_feats3))
    
    spectral = MultiviewSpectralClustering(n_clusters, n_views=n_views, random_state=random_seed)
    return {'n_fit' : num_fit_samples, 'n_feats1': n_feats1, 'n_feats2': n_feats2,
            'n_feats3' : n_feats3, 'n_clusters': n_clusters, 'spectral' : spectral,
            'fit_data' : fit_data, 'n_views' : n_views}

def test_gaussian_sim(data):

    v1_data = data['fit_data'][0]
    distances = cdist(v1_data, v1_data)
    gamma = 1/ (2 * np.median(distances) **2)
    true_kernel = rbf_kernel(v1_data, v1_data, gamma)
    spectral = data['spectral']
    g_kernel = spectral._gaussian_sim(v1_data)

    assert(g_kernel.shape[0] == data['n_fit'])
    assert(g_kernel.shape[1] == data['n_fit'])
    
    for ind1 in range(g_kernel.shape[0]):
        for ind2 in range(g_kernel.shape[1]):
            assert np.abs(true_kernel[ind1][ind2]
                          - g_kernel[ind1][ind2]) < 0.000001
    
def test_compute_eigs(data):

    v1_data = data['fit_data'][0]
    distances = cdist(v1_data, v1_data)
    gamma = 1/ (2 * np.median(distances) **2)
    g_kernel = rbf_kernel(v1_data, v1_data, gamma)
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
    spectral = MultiviewSpectralClustering(2)
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

        

def test_fit_predict_info_view(data):
    
    v_data = data['fit_data']
    n_views = data['n_views']
    n_clusts = data['n_clusters']
    info_view = np.random.randint(n_views)
    spectral = MultiviewSpectralClustering(n_clusts, n_views=n_views, info_view=info_view)
    predictions = spectral.fit_predict(v_data) 
    
    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)

        
def test_fit_predict_n_iter(data):

    v_data = data['fit_data']
    spectral = data['spectral']
    n_iter = 5
    predictions = spectral.fit_predict(v_data, n_iter=n_iter)
    n_clusts = data['n_clusters']
    
    assert(predictions.shape[0] == data['n_fit'])
    for clust in predictions:
        assert(clust >= 0 and clust < n_clusts)

