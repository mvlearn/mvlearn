# KCCA Unit Tests

from mvlearn.embed.kcca import KCCA, _center_norm, _make_kernel, _make_icd_kernel
import numpy as np
import pytest

# Initialize number of samples
nSamples = 1000
np.random.seed(30)

# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(nSamples,)
latvar2 = np.random.randn(nSamples,)

# Define independent components for each dataset (number of observations x dataset dimensions)
indep1 = np.random.randn(nSamples, 4)
indep2 = np.random.randn(nSamples, 5)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
data1 = 0.25*indep1 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2)).T
data2 = 0.25*indep2 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

# Split each dataset into a training set and test set (10% of dataset is training data)
train1 = data1[:int(nSamples/10)]
train2 = data2[:int(nSamples/10)]
test1 = data1[int(nSamples/10):]
test2 = data2[int(nSamples/10):]

n_components = 4

# Initialize a linear kCCA class
kcca_l = KCCA(ktype ="linear", reg = 0.001, n_components = n_components)

# Use the methods to find a kCCA mapping and transform the views of data
kcca_ft = kcca_l.fit_transform([train1, train2])
kcca_f = kcca_l.fit([train1, train2])
kcca_t = kcca_l.transform([train1, train2])

#gaussian related data
N = 100
t = np.random.uniform(-np.pi, np.pi, N)
e1 = np.random.normal(0, 0.05, (N,2))
e2 = np.random.normal(0, 0.05, (N,2))

x = np.zeros((N,2))
x[:,0] = t
x[:,1] = np.sin(3*t)
x += e1

y = np.zeros((N,2))
y[:,0] = np.exp(t/4)*np.cos(2*t)
y[:,1] = np.exp(t/4)*np.sin(2*t)
y += e2

# Test that number of components is equal to n_components
def test_numCC_components_():
    assert len(kcca_ft[0][0]) and len(kcca_ft[1][0]) == n_components

# Test that number of views is equal to number of ws_
def test_numCC_ws_():
    assert len(kcca_f.weights_) == 2
    
# Test that number of views is equal to number of comps_
def test_numCC_comps_():
    assert len(kcca_ft) == 2

    
# Test that components from transform equals fit.transform weights
def test_ktype_components():
    assert np.allclose(kcca_ft, kcca_t)
    
# Test that gaussian kernel runs
def test_ktype_gaussian():
    kgauss = KCCA(ktype = 'gaussian', reg = 0.0001, n_components = 2, sigma=2.0)
    a = kgauss.fit_transform([train1, train2])
    assert len(a) == 2
    
# Test that polynomial kernel runs
def test_ktype_polynomial():
    kpoly = KCCA(ktype = 'poly', reg = 0.0001, n_components = 2, degree=3.0)
    b = kpoly.fit_transform([train1, train2])
    assert len(b) == 2

### Testing helper functions
np.random.seed(30)
a = np.random.uniform(0,1,(10,10))
b = np.random.uniform(0,1,(10,10))
c = np.ones((10,3))
c[1] = 0
    
# Test that center_nrom works
def test_center_norm():
    b = _center_norm(a)
    assert np.allclose(np.mean(b),0)
    
# Test make_kernel
def test_make_kernel(): 
    lkernel = _make_kernel(c, c, ktype="linear")
    gkernel = _make_kernel(c, c, ktype="gaussian", sigma=1.0, degree=2.0)
    pkernel = _make_kernel(c, c, ktype="poly", sigma=1.0, degree=2.0)
    assert lkernel.shape==gkernel.shape==pkernel.shape == (10,10)
    
# Test error handling
def test_bad_ktype():
    with pytest.raises(ValueError):
        kcca_a = KCCA(ktype ="test", reg = 0.001, n_components = n_components)

# Test error handling
def test_neg_nc():
    with pytest.raises(ValueError):
        kcca_b = KCCA(ktype ="linear", reg = 0.001, n_components = -1)
        
# Test error handling
def test_float_nc():
    with pytest.raises(ValueError):
        kcca_c = KCCA(ktype ="linear", reg = 0.001, n_components = 1.0)
  
# Test error handling
def test_neg_reg():
    with pytest.raises(ValueError):
        kcca_d = KCCA(ktype ="linear", reg = -0.001, n_components = 1)
   
# Test error handling
def test_neg_sigma():
    with pytest.raises(ValueError):
        kcca_e = KCCA(ktype ="gaussian", reg = 0.001, n_components = 1, sigma =-1.0)

# Test error handling
def test_neg_degree():
    with pytest.raises(ValueError):
        kcca_f = KCCA(ktype ="poly", reg = 0.001, n_components = 1, degree =-1)
        
# Test error handling
def test_neg_constant():
    with pytest.raises(ValueError):
        kcca_g = KCCA(ktype ="poly", reg = 0.001, constant = -1)
        
# Test if error when transform before fit
def test_no_weights():
    with pytest.raises(NameError):
        kcca_b = KCCA(ktype ="linear", reg = 0.001, n_components = 1)
        kcca_b.transform([train1, train2])
        
# Test mrank error
def test_mrank_neg():
    with pytest.raises(ValueError):
        kcca_z = KCCA(ktype ="poly", decomp = "icd", reg = 0.001, mrank = -1)

# Test method error
def test_method():
    with pytest.raises(ValueError):
        kcca_v = KCCA(ktype ="poly", method = "test", reg = 0.001, mrank = 5)

# Test precision error
def test_precision():
    with pytest.raises(ValueError):
        kcca_x = KCCA(ktype ="poly", decomp = "icd", reg = 0.001, precision = -1)

# Test that _zscore works
def test_icd_mrank():
    kcca_g_icd = KCCA(ktype ="gaussian", sigma = 1.0, n_components = 2, reg = 0.01, decomp = 'icd', mrank = 2)
    icd = kcca_g_icd.fit_transform([x, y])
    assert (len(icd) == 2)

# Test make_icd_kernel
def test_make_icd_kernelg(): 
    g_icd_kernel = _make_icd_kernel(x, ktype="gaussian", mrank = 2)
    l_icd_kernel = _make_icd_kernel(x, ktype="linear", constant=1, mrank = 2)
    p_icd_kernel = _make_icd_kernel(x, ktype="poly", degree = 3, mrank = 2)
    assert g_icd_kernel.shape==l_icd_kernel.shape==p_icd_kernel.shape == (100,2)

# Test getting stats wrongly
def test_get_stats_wrong():
    kcca_bad = KCCA()
    with pytest.raises(NameError):
        kcca_bad.get_stats()
    with pytest.raises(NameError):
        kcca_bad.fit([train1, train2])
        stats = kcca_bad.get_stats()

def test_get_stats_nonlinear_kernel():
    kcca_poly = KCCA(ktype='poly')
    kcca_poly.fit([train1, train2]).transform([train1, train2])
    stats = kcca_poly.get_stats()
    assert np.all(stats['r']>0)
    assert stats['r'].shape == (2,)

    kcca_gaussian = KCCA(ktype='gaussian')
    kcca_gaussian.fit([train1, train2]).transform([train1, train2])
    stats = kcca_gaussian.get_stats()
    assert np.all(stats['r']>0)
    assert stats['r'].shape == (2,)

def test_get_stats_icd_check_corrs():
    X = np.vstack((np.eye(3,3), 2*np.eye(3,3)))
    Y1 = np.fliplr(np.eye(3,3))
    Y = np.vstack((Y1, 0.1*np.eye(3,3)))

    kcca = KCCA(n_components=3, decomp='icd')
    out = kcca.fit([X, Y]).transform([X, Y])
    stats = kcca.get_stats()

    assert np.allclose(stats['r'], np.array([0.51457091, 0.3656268]))

# Test getting stats correctly, and check against stats that Matlab canoncorr gives
def test_get_stats_vs_matlab():
    X = np.vstack((np.eye(3,3), 2*np.eye(3,3)))
    Y1 = np.fliplr(np.eye(3,3))
    Y = np.vstack((Y1, 0.1*np.eye(3,3)))
    matlab_stats = {'r': np.array([1.000000000000000, 0.533992991387982, 0.355995327591988]),
                    'Wilks': np.array([0, 0.624256445446525, 0.873267326732673]),
                    'df1': np.array([9, 4, 1]),
                    'df2': np.array([0.150605850666856, 2, 2]),
                    'F': np.array([np.inf, 0.132832080200501, 0.290249433106576]),
                    'pF': np.array([0, 0.955941574355455, 0.644004672408012]),
                    'chisq': np.array([np.inf, 0.706791037156489, 0.542995281660087]),
                    'pChisq': np.array([0, 0.950488814632803, 0.461194028737338])
                    }

    kcca = KCCA(n_components=3)
    out = kcca.fit([X, Y]).transform([X, Y])
    stats = kcca.get_stats()

    assert np.allclose(stats['r'][0], 1)
    nondegen = np.argwhere(stats['r'] < 1 - 2 * np.finfo(float).eps).squeeze()
    assert np.array_equal(nondegen, np.array([1, 2]))

    for key in stats:
        assert np.allclose(stats[key], matlab_stats[key], rtol=1e-3, atol=1e-4)

def test_get_stats_1_feature_vs_matlab():
    X = np.arange(1, 11).reshape(-1, 1)
    Y = np.arange(2, 21, 2).reshape(-1, 1)
    matlab_stats = {'r': np.array([1]),
                    'Wilks': np.array([0]),
                    'df1': np.array([1]),
                    'df2': np.array([8]),
                    'F': np.array([np.inf]),
                    'pF': np.array([0]),
                    'chisq': np.array([np.inf]),
                    'pChisq': np.array([0])
                    }

    kcca = KCCA(n_components=1)
    out = kcca.fit([X, Y]).transform([X, Y])
    stats = kcca.get_stats()

    for key in stats:
        assert np.allclose(stats[key], matlab_stats[key], rtol=1e-3, atol=1e-4)

def test_get_stats_1_component():
    np.random.seed(12)
    X = X = np.random.rand(100,3)
    Y = np.random.rand(100,4)
    past_stats = {'r': np.array([0.22441608326082138]),
                    'Wilks': np.array([0.94963742]),
                    'df1': np.array([12]),
                    'df2': np.array([246.34637455]),
                    'F': np.array([0.40489714]),
                    'pF': np.array([0.96096493]),
                    'chisq': np.array([4.90912773]),
                    'pChisq': np.array([0.9609454])
                    }

    kcca1 = KCCA(n_components=1)
    kcca1.fit_transform([X,Y])
    stats = kcca1.get_stats()

    assert not stats['r'] == 1
    assert not stats['r'] + 2 * np.finfo(float).eps >= 1

    for key in stats:
        assert np.allclose(stats[key], past_stats[key], rtol=1e-3, atol=1e-4)

def test_get_stats_2_components():
    np.random.seed(12)
    X = X = np.random.rand(100,3)
    Y = np.random.rand(100,4)
    past_stats = {'r': np.array([0.22441608, 0.19056307]),
                    'Wilks': np.array([0.91515202, 0.96368572]),
                    'df1': np.array([12, 6]),
                    'df2': np.array([246.34637455, 188]),
                    'F': np.array([0.69962605, 0.58490315]),
                    'pF': np.array([0.75134965, 0.74212361]),
                    'chisq': np.array([8.42318331, 4.2115406 ]),
                    'pChisq': np.array([0.75124771, 0.64807349])
                    }

    kcca2 = KCCA(n_components=2)
    kcca2.fit_transform([X,Y])
    stats = kcca2.get_stats()

    nondegen = np.argwhere(stats['r'] < 1 - 2 * np.finfo(float).eps).squeeze()
    assert np.array_equal(nondegen, np.array([0, 1]))

    for key in stats:
        assert np.allclose(stats[key], past_stats[key], rtol=1e-3, atol=1e-4)



