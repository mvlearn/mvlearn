import pytest
from mvlearn.datasets import GaussianMixture
from numpy.testing import assert_equal
import numpy as np


n = 100
gm_uni = GaussianMixture([0,1], np.eye(2), n)
mu = [[-1,0], [1,0]]
sigma = [[[1,0],[0,1]], [[1,0],[1,2]]]
class_probs = [0.3, 0.7]
gm_multi = GaussianMixture(mu, sigma, n, class_probs)

def test_multivariate():
    latents, _ = gm_multi.get_Xy(latents=True)
    assert_equal(n, len(latents))
    assert_equal(len(mu[0]), latents.shape[1])

def test_class_probs():
    _, y = gm_multi.get_Xy(latents=True)
    for i,p in enumerate(class_probs):
        assert_equal(int(p*n), list(y).count(i))

def test_transforms():
    transforms = ['linear', 'poly', 'sin', lambda x: 2*x+1]
    for transform in transforms:
        gm_uni.sample_views(transform, n_noise=2)
        assert_equal(len(gm_uni.get_Xy()[0]), 2)
        assert_equal(gm_uni.get_Xy()[0][0].shape, (n,4))
        assert_equal(gm_uni.get_Xy()[0][1].shape, (n,4))

def test_bad_class_probs():
    with pytest.raises(ValueError):
        GaussianMixture(mu, sigma, n, class_probs=[0.3, 0.4])

def test_bad_transform_function():
    with pytest.raises(TypeError):
        gm_uni.sample_views(list())

def test_bad_transform_string():
    with pytest.raises(ValueError):
        gm_uni.sample_views('error')

def test_no_sample():
    gaussm = GaussianMixture(mu, sigma, n, class_probs)
    with pytest.raises(NameError):
        gaussm.get_Xy()

def test_bad_shapes():
    ## Wrong Length
    with pytest.raises(ValueError):
        GaussianMixture([1], sigma, n)
    ## Inconsistent dimension
    with pytest.raises(ValueError):
        GaussianMixture(mu, [np.eye(2), np.eye(3)], n, class_probs)
    ## Wrong uni dimensions
    with pytest.raises(ValueError):
        GaussianMixture([1,0], [1,0], n)
    ## Wrong multi sizes
    with pytest.raises(ValueError):
        GaussianMixture(mu, sigma, n, class_probs=[0.3, 0.1, 0.6])

def test_random_state():
    gm_1 = GaussianMixture(mu, sigma, 10, class_probs, random_state=42)
    gm_1.sample_views('poly')
    Xs_1, y_1 = gm_1.get_Xy()
    gm_2 = GaussianMixture(mu, sigma, 10, class_probs, random_state=42)
    gm_2.sample_views('poly')
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)

def test_shuffle():
    gm_1 = GaussianMixture(mu, sigma, 500, class_probs, random_state=42,
                           shuffle=True)
    gm_1.sample_views('poly')
    Xs_1, y_1 = gm_1.get_Xy()
    gm_2 = GaussianMixture(mu, sigma, 500, class_probs, random_state=42,
                           shuffle=True)
    gm_2.sample_views('poly')
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert not np.allclose(view1, view2)
    assert not np.allclose(y_1, y_2)

def test_shuffle_with_random_state():
    gm_1 = GaussianMixture(mu, sigma, 500, class_probs, random_state=42,
                           shuffle=True, shuffle_random_state=42)
    gm_1.sample_views('poly')
    Xs_1, y_1 = gm_1.get_Xy()
    gm_2 = GaussianMixture(mu, sigma, 500, class_probs, random_state=42,
                           shuffle=True, shuffle_random_state=42)
    gm_2.sample_views('poly')
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)

test_multivariate()
test_class_probs()
test_transforms()
test_bad_class_probs()
test_bad_transform_function()
test_bad_transform_string()
test_no_sample()
test_bad_shapes()
test_random_state()
test_shuffle()
test_shuffle_with_random_state()
