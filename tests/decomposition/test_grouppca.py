import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from mvlearn.decomposition.grouppca import GroupPCA
import scipy


@pytest.mark.parametrize("n_components", [None, 1, 3, 9])
@pytest.mark.parametrize("n_individual_components",
                         ['auto', None, 3, [2, 3, 4]])
def test_pca(n_components, n_individual_components):
    gpca = GroupPCA(
        n_components=n_components,
        n_individual_components=n_individual_components
    )
    n_samples = 100
    n_features = [6, 4, 5]
    rng = np.random.RandomState(0)
    Xs = [
        rng.multivariate_normal(np.zeros(p), np.eye(p), size=n_samples)
        for p in n_features
    ]
    # check the shape of fit.transform
    X_r = gpca.fit(Xs).transform(Xs)
    assert X_r.shape[0] == n_samples
    if n_components is not None:
        assert X_r.shape[1] == n_components

    # check the equivalence of fit.transform and fit_transform
    X_r2 = gpca.fit_transform(Xs)
    assert_allclose(X_r, X_r2)
    X_r = gpca.transform(Xs)
    assert_allclose(X_r, X_r2)


@pytest.mark.parametrize("n_individual_components", [None, 3, [2, 3, 4]])
def test_whitening(n_individual_components, individual_output):
    # Check that PCA output has unit-variance
    rng = np.random.RandomState(0)
    n_samples = 100
    n_features = 80
    n_components = 30
    rank = 50

    # some low rank data with correlated features
    X = np.dot(
        rng.randn(n_samples, rank),
        np.dot(
            np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)
        ),
    )
    # the component-wise variance of the first 50 features is 3 times the
    # mean component-wise variance of the remaining 30 features
    X[:, :50] *= 3
    assert X.shape == (n_samples, n_features)
    # the component-wise variance is thus highly varying:
    assert X.std(axis=0).std() > 43.8
    Xs = np.array_split(X, 3, axis=1)
    # whiten the data while projecting to the lower dim subspace
    Xs_ = Xs.copy()  # make sure we keep an original across iterations.
    gpca = GroupPCA(
        n_components=n_components,
        whiten=True,
        random_state=0,
        n_individual_components=n_individual_components,
    )
    # test fit_transform
    X_whitened = gpca.fit_transform(Xs_)
    X_whitened2 = gpca.transform(Xs_)
    assert_allclose(X_whitened, X_whitened2, rtol=5e-4)
    assert X_whitened.shape == (n_samples, n_components)
    assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components))

    Xs_ = Xs.copy()
    gpca = GroupPCA(
        n_components=n_components,
        whiten=False,
        n_individual_components=n_individual_components,
    ).fit(Xs)
    X_unwhitened = gpca.transform(Xs_)
    assert X_unwhitened.shape == (n_samples, n_components)
    # in that case the output components still have varying variances
    assert X_unwhitened.std(axis=0).std() > 20


@pytest.mark.parametrize("whiten", [False, True])
@pytest.mark.parametrize("n_individual_components", [None, 2, [2, 2]])
def test_grouppca_inverse(n_individual_components, whiten):
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= 0.00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    X2 = np.copy(X)
    X2[:, 1] += rng.randn(n) * 0.00001

    Xs = [X, X2]

    gpca = GroupPCA(
        n_components=2,
        whiten=whiten,
        n_individual_components=n_individual_components,
    ).fit(Xs)
    Y = gpca.transform(Xs)
    Y_inverse = gpca.inverse_transform(Y)
    assert len(Y_inverse) == len(Xs)
    for X, X_estimated in zip(Xs, Y_inverse):
        assert_allclose(X, X_estimated, atol=1e-4)


def test_grouppca_deterministic_output():
    n_samples = 100
    n_features = [6, 4, 5]
    rng = np.random.RandomState(0)
    Xs = [
        rng.multivariate_normal(np.zeros(p), np.eye(p), size=n_samples)
        for p in n_features
    ]
    transformed_X = np.zeros((20, 2))
    for i in range(20):
        pca = GroupPCA(
            n_components=2, n_individual_components=3, random_state=rng
        )
        transformed_X[i, :] = pca.fit_transform(Xs)[0]
    assert_allclose(
        transformed_X, np.tile(transformed_X[0, :], 20).reshape(20, 2)
    )


def test_grouppca_transform():
    X1 = np.zeros((100, 2))
    X1[:, 0] = np.arange(100)
    X1[:, 1] = 5 * np.arange(100) + np.random.rand(100)
