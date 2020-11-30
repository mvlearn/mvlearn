import pytest
from mvlearn.datasets import make_gaussian_mixture
from numpy.testing import assert_equal
import numpy as np


n_samples = 100
centers = [[-1, 0], [1, 0]]
covariances = [[[1, 0], [0, 1]], [[1, 0], [1, 2]]]
class_probs = [0.3, 0.7]
Xs, y, latents = make_gaussian_mixture(
    n_samples, centers, covariances, class_probs=class_probs,
    return_latents=True)


def test_formats():
    assert_equal(n_samples, len(latents))
    assert_equal(len(centers[0]), latents.shape[1])
    assert_equal(Xs[0], latents)


def test_class_probs():
    for i, p in enumerate(class_probs):
        assert_equal(int(p * n_samples), list(y).count(i))


@pytest.mark.parametrize(
    "transform", ["linear", "poly", "sin", lambda x: 2 * x + 1])
def test_transforms(transform):
    Xs, y, latents = make_gaussian_mixture(
        n_samples, centers, covariances, class_probs=class_probs,
        return_latents=True, transform=transform, noise_dims=2)

    assert_equal(len(Xs), 2)
    assert_equal(Xs[0].shape, (n_samples, 4))
    assert_equal(Xs[1].shape, (n_samples, 4))


def test_bad_class_probs():
    with pytest.raises(ValueError):
        make_gaussian_mixture(
            centers, covariances, n_samples, class_probs=[0.3, 0.4]
        )


@pytest.mark.parametrize(
    "transform", [list(), None])
def test_bad_transform_value(transform):
    with pytest.raises(TypeError):
        make_gaussian_mixture(
            n_samples, centers, covariances, transform=transform)


@pytest.mark.parametrize(
    "transform", ["error"])
def test_bad_transform_type(transform):
    with pytest.raises(ValueError):
        make_gaussian_mixture(
            n_samples, centers, covariances, transform=transform)


def test_bad_shapes():
    # Wrong Length
    with pytest.raises(ValueError):
        make_gaussian_mixture(n_samples, [1], covariances)
    # Inconsistent dimension
    with pytest.raises(ValueError):
        make_gaussian_mixture(
            n_samples, centers, [np.eye(2), np.eye(3)],
            class_probs=class_probs
        )
    # Wrong uni dimensions
    with pytest.raises(ValueError):
        make_gaussian_mixture(n_samples, [1, 0], [1, 0])
    # Wrong centerslti sizes
    with pytest.raises(ValueError):
        make_gaussian_mixture(
            n_samples, centers, covariances, class_probs=[0.3, 0.1, 0.6]
        )


@pytest.mark.parametrize("noise", [None, 0, 1])
def test_random_state(noise):
    Xs_1, y_1 = make_gaussian_mixture(
        10, centers, covariances, class_probs=class_probs,
        transform='poly', random_state=42, noise=noise
    )
    Xs_2, y_2 = make_gaussian_mixture(
        10, centers, covariances, class_probs=class_probs,
        transform='poly', random_state=42, noise=noise
    )
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)


def test_noise_dims_not_same_but_reproducible():
    Xs_1, _ = make_gaussian_mixture(
        20, centers, covariances, class_probs=class_probs, random_state=42,
        transform="poly", noise_dims=2
    )
    view1_noise, view2_noise = Xs_1[0][:, -2:], Xs_1[1][:, -2:]
    assert not np.allclose(view1_noise, view2_noise)
    Xs_2, _ = make_gaussian_mixture(
        20, centers, covariances, class_probs=class_probs, random_state=42,
        transform="poly", noise_dims=2
    )
    view1_noise2, view2_noise2 = Xs_2[0][:, -2:], Xs_2[1][:, -2:]
    assert np.allclose(view1_noise, view1_noise2)
    assert np.allclose(view2_noise, view2_noise2)


@pytest.mark.parametrize(
    "transform", ["linear", "poly", "sin", lambda x: 2 * x + 1])
def test_signal_noise_not_same_but_reproducible(transform):
    Xs_1, _ = make_gaussian_mixture(
        20, centers, covariances, class_probs=class_probs, random_state=42,
        transform=transform, noise=1
    )
    view1_noise, view2_noise = Xs_1[0], Xs_1[1]
    Xs_2, _ = make_gaussian_mixture(
        20, centers, covariances, class_probs=class_probs, random_state=42,
        transform=transform, noise=1
    )
    view1_noise2, view2_noise2 = Xs_2[0], Xs_2[1]
    # Noise is reproducible and signal is the same
    assert np.allclose(view1_noise, view1_noise2)
    assert np.allclose(view2_noise, view2_noise2)
    Xs_3, _ = make_gaussian_mixture(
        20, centers, covariances, class_probs=class_probs, random_state=42,
        transform=transform
    )
    view1_noise3, view2_noise3 = Xs_3[0], Xs_3[1]
    # Noise varies view1, but keeps view 2 unaffects (i.e. the latents)
    assert not np.allclose(view1_noise, view1_noise3)
    assert np.allclose(view2_noise, view2_noise3)


def test_shuffle():
    np.random.seed(42)
    Xs_1, y_1 = make_gaussian_mixture(
        20,
        centers,
        covariances,
        class_probs=class_probs,
        transform='poly',
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    np.random.seed(30)
    Xs_2, y_2 = make_gaussian_mixture(
        20,
        centers,
        covariances,
        class_probs=class_probs,
        transform='poly',
        random_state=42,
        shuffle=True,
        shuffle_random_state=10,
    )
    for view1, view2 in zip(Xs_1, Xs_2):
        assert not np.allclose(view1, view2)
    assert not np.allclose(y_1, y_2)


def test_shuffle_with_random_state():
    Xs_1, y_1 = make_gaussian_mixture(
        20,
        centers,
        covariances,
        class_probs=class_probs,
        transform='poly',
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    Xs_2, y_2 = make_gaussian_mixture(
        20,
        centers,
        covariances,
        class_probs=class_probs,
        transform='poly',
        random_state=42,
        shuffle=True,
        shuffle_random_state=42,
    )
    for view1, view2 in zip(Xs_1, Xs_2):
        assert np.allclose(view1, view2)
    assert np.allclose(y_1, y_2)
