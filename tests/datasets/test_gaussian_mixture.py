import pytest
from mvlearn.datasets import GaussianMixture
from numpy.testing import assert_equal
import numpy as np


n_samples = 100
gm_uni = GaussianMixture(n_samples, [0, 1], np.eye(2))
centers = [[-1, 0], [1, 0]]
covariances = [[[1, 0], [0, 1]], [[1, 0], [1, 2]]]
class_probs = [0.3, 0.7]
gm_centerslti = GaussianMixture(n_samples, centers, covariances, class_probs)


def test_centersltivariate():
    latents, _ = gm_centerslti.get_Xy(latents=True)
    assert_equal(n_samples, len(latents))
    assert_equal(len(centers[0]), latents.shape[1])


def test_class_probs():
    _, y = gm_centerslti.get_Xy(latents=True)
    for i, p in enumerate(class_probs):
        assert_equal(int(p * n_samples), list(y).count(i))


def test_transforms():
    transforms = ["linear", "poly", "sin", lambda x: 2 * x + 1]
    for transform in transforms:
        gm_uni.sample_views(transform, n_noise=2)
        assert_equal(len(gm_uni.get_Xy()[0]), 2)
        assert_equal(gm_uni.get_Xy()[0][0].shape, (n_samples, 4))
        assert_equal(gm_uni.get_Xy()[0][1].shape, (n_samples, 4))


def test_bad_class_probs():
    with pytest.raises(ValueError):
        GaussianMixture(
            centers, covariances, n_samples, class_probs=[0.3, 0.4]
        )


def test_bad_transform_function():
    with pytest.raises(TypeError):
        gm_uni.sample_views(list())


def test_bad_transform_string():
    with pytest.raises(ValueError):
        gm_uni.sample_views("error")


def test_no_sample():
    gaussm = GaussianMixture(n_samples, centers, covariances, class_probs)
    with pytest.raises(NameError):
        gaussm.get_Xy()


def test_bad_shapes():
    ## Wrong Length
    with pytest.raises(ValueError):
        GaussianMixture(n_samples, [1], covariances)
    ## Inconsistent dimension
    with pytest.raises(ValueError):
        GaussianMixture(
            n_samples, centers, [np.eye(2), np.eye(3)], class_probs
        )
    ## Wrong uni dimensions
    with pytest.raises(ValueError):
        GaussianMixture(n_samples, [1, 0], [1, 0])
    ## Wrong centerslti sizes
    with pytest.raises(ValueError):
        GaussianMixture(
            n_samples, centers, covariances, class_probs=[0.3, 0.1, 0.6]
        )


def test_random_state():
    gm_1 = GaussianMixture(
        10, centers, covariances, class_probs, random_state=42
    )
    gm_1.sample_views("poly")
    Xs_1, y_1 = gm_1.get_Xy()
    gm_2 = GaussianMixture(
        10, centers, covariances, class_probs, random_state=42
    )
    gm_2.sample_views("poly")
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)


def test_noise_dims_not_same_but_reproducible():
    gm_1 = GaussianMixture(
        20, centers, covariances, class_probs, random_state=42
    )
    gm_1.sample_views("poly", n_noise=2)
    Xs_2, _ = gm_1.get_Xy()
    view1_noise, view2_noise = Xs_2[0][:, -2:], Xs_2[1][:, -2:]
    assert not np.allclose(view1_noise, view2_noise)
    gm_2 = GaussianMixture(
        20, centers, covariances, class_probs, random_state=42
    )
    gm_2.sample_views("poly", n_noise=2)
    Xs_2, _ = gm_1.get_Xy()
    view1_noise2, view2_noise2 = Xs_2[0][:, -2:], Xs_2[1][:, -2:]
    assert np.allclose(view1_noise, view1_noise2)
    assert np.allclose(view2_noise, view2_noise2)


def test_shuffle():
    np.random.seed(42)
    gm_1 = GaussianMixture(
        20,
        centers,
        covariances,
        class_probs,
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    gm_1.sample_views("poly")
    Xs_1, y_1 = gm_1.get_Xy()
    np.random.seed(30)
    gm_2 = GaussianMixture(
        20,
        centers,
        covariances,
        class_probs,
        random_state=42,
        shuffle=True,
        shuffle_random_state=10,
    )
    gm_2.sample_views("poly")
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert not np.allclose(view1, view2)
    assert not np.allclose(y_1, y_2)


def test_shuffle_with_random_state():
    gm_1 = GaussianMixture(
        20,
        centers,
        covariances,
        class_probs,
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    gm_1.sample_views("poly")
    Xs_1, y_1 = gm_1.get_Xy()
    gm_2 = GaussianMixture(
        20,
        centers,
        covariances,
        class_probs,
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    gm_2.sample_views("poly")
    Xs_2, y_2 = gm_2.get_Xy()
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)


test_centersltivariate()
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
