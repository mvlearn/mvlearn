# CCA Unit Tests

from mvlearn.embed import CCA
import numpy as np
import pytest

# Initialize number of samples
nSamples = 1000
np.random.seed(30)

# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(nSamples,)
latvar2 = np.random.randn(nSamples,)

# Define independent components for each dataset
# #(number of observations x dataset dimensions)
indep1 = np.random.randn(nSamples, 4)
indep2 = np.random.randn(nSamples, 5)

# Create two datasets, with each dimension composed as a sum of 75% one
# of the latent variables and 25% independent component
data1 = 0.25*indep1 + 0.75*np.vstack(
    (latvar1, latvar2, latvar1, latvar2)).T
data2 = 0.25*indep2 + 0.75*np.vstack(
    (latvar1, latvar2, latvar1, latvar2, latvar1)).T

# Split each dataset into a training set and test set
# (10% of dataset is training data)
train1 = data1[:int(nSamples/10)]
train2 = data2[:int(nSamples/10)]
test1 = data1[int(nSamples/10):]
test2 = data2[int(nSamples/10):]

n_components = 4

# Initialize a linear kCCA class
cca = CCA(regs=0.001, n_components=n_components)

# Use the methods to find a kCCA mapping and transform the views of data
cca_ft = cca.fit_transform([train1, train2])
cca_f = cca.fit([train1, train2])
cca_t = cca.transform([train1, train2])

# gaussian related data
N = 100
t = np.random.uniform(-np.pi, np.pi, N)
e1 = np.random.normal(0, 0.05, (N, 2))
e2 = np.random.normal(0, 0.05, (N, 2))

x = np.zeros((N, 2))
x[:, 0] = t
x[:, 1] = np.sin(3*t)
x += e1

y = np.zeros((N, 2))
y[:, 0] = np.exp(t/4)*np.cos(2*t)
y[:, 1] = np.exp(t/4)*np.sin(2*t)
y += e2


# Test that number of components is equal to n_components
def test_numCC_components_():
    assert len(cca_ft[0][0]) and len(cca_ft[1][0]) == n_components


# Test that number of views is equal to number of ws_
def test_numCC_ws_():
    assert len(cca_f.loadings_) == 2


# Test that number of views is equal to number of comps_
def test_numCC_comps_():
    assert len(cca_ft) == 2


# Test that components from transform equals fit.transform weights
def test_ktype_components():
    assert np.allclose(cca_ft, cca_t)


# Test get_stats() fail cases
def test_errors():
    cca = CCA()
    with pytest.raises(ValueError):
        cca.fit([train1, train2, train2])

    with pytest.raises(ValueError):
        cca.fit([train1, train2])
        cca.get_stats([train1, train1, train1])

    with pytest.raises(AssertionError):
        cca.fit([train1, train2])
        _ = cca.get_stats([train1, train2])

    with pytest.raises(KeyError):
        cca.fit([train1, train2])
        _ = cca.get_stats([train1, train1], 'FAIL')


# Test getting stats correctly, and check against stats that
# Matlab canoncorr gives
def test_get_stats_vs_matlab():
    X = np.vstack((np.eye(3, 3), 2*np.eye(3, 3)))
    Y1 = np.fliplr(np.eye(3, 3))
    Y = np.vstack((Y1, 0.1*np.eye(3, 3)))
    matlab_stats = {
        'r': np.array(
            [1.000000000000000, 0.533992991387982, 0.355995327591988]),
        'Wilks': np.array([0, 0.624256445446525, 0.873267326732673]),
        'df1': np.array([9, 4, 1]),
        'df2': np.array([0.150605850666856, 2, 2]),
        'F': np.array([np.inf, 0.132832080200501, 0.290249433106576]),
        'pF': np.array([0, 0.955941574355455, 0.644004672408012]),
        'chisq': np.array([np.inf, 0.706791037156489, 0.542995281660087]),
        'pChisq': np.array([0, 0.950488814632803, 0.461194028737338])
        }

    cca = CCA(n_components=3)
    scores = cca.fit_transform([X, Y])
    stats = cca.get_stats(scores)

    assert np.allclose(stats['r'][0], 1)
    nondegen = np.argwhere(stats['r'] < 1 - 2 * np.finfo(float).eps).squeeze()
    assert np.array_equal(nondegen, np.array([1, 2]))

    for key in stats:
        assert np.allclose(stats[key], matlab_stats[key], rtol=1e-3, atol=1e-4)


def test_get_stats_1_feature_vs_matlab():
    X = np.arange(1, 11).reshape(-1, 1)
    Y = np.arange(2, 21, 2).reshape(-1, 1)
    matlab_stats = {
        'r': np.array([1]),
        'Wilks': np.array([0]),
        'df1': np.array([1]),
        'df2': np.array([8]),
        'F': np.array([np.inf]),
        'pF': np.array([0]),
        'chisq': np.array([np.inf]),
        'pChisq': np.array([0])
        }

    cca = CCA(n_components=1)
    scores = cca.fit_transform([X, Y])
    stats = cca.get_stats(scores)

    for key in stats:
        assert np.allclose(stats[key], matlab_stats[key], rtol=1e-3, atol=1e-4)


def test_get_stats_1_component():
    np.random.seed(12)
    X = X = np.random.rand(100, 3)
    Y = np.random.rand(100, 4)
    past_stats = {
        'r': np.array([0.22441608326082138]),
        'Wilks': np.array([0.94963742]),
        'df1': np.array([12]),
        'df2': np.array([246.34637455]),
        'F': np.array([0.40489714]),
        'pF': np.array([0.96096493]),
        'chisq': np.array([4.90912773]),
        'pChisq': np.array([0.9609454])
        }

    cca = CCA(n_components=1)
    scores = cca.fit_transform([X, Y])
    stats = cca.get_stats(scores)

    assert not stats['r'] == 1
    assert not stats['r'] + 2 * np.finfo(float).eps >= 1

    for key in stats:
        assert np.allclose(stats[key], past_stats[key], rtol=1e-3, atol=1e-4)


def test_get_stats_2_components():
    np.random.seed(12)
    X = X = np.random.rand(100, 3)
    Y = np.random.rand(100, 4)
    past_stats = {
        'r': np.array([0.22441608, 0.19056307]),
        'Wilks': np.array([0.91515202, 0.96368572]),
        'df1': np.array([12, 6]),
        'df2': np.array([246.34637455, 188]),
        'F': np.array([0.69962605, 0.58490315]),
        'pF': np.array([0.75134965, 0.74212361]),
        'chisq': np.array([8.42318331, 4.2115406]),
        'pChisq': np.array([0.75124771, 0.64807349])
        }

    cca = CCA(n_components=2)
    scores = cca.fit_transform([X, Y])
    stats = cca.get_stats(scores)

    nondegen = np.argwhere(stats['r'] < 1 - 2 * np.finfo(float).eps).squeeze()
    assert np.array_equal(nondegen, np.array([0, 1]))

    for key in stats:
        assert np.allclose(stats[key], past_stats[key], rtol=1e-3, atol=1e-4)
