# KCCA Unit Tests

from mvlearn.embed.kcca import KCCA, _center_norm, _make_kernel
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
        
    