# KCCA Unit Tests

from multiview.embed.kcca import KCCA, _zscore, _rowcorr, _listdot, _listcorr, _make_kernel
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

# Test that cancorrs_ is equal to n_components
def test_numCC_cancorrs_():
    assert len(kcca_ft.cancorrs_) == n_components

# Test that number of views is equal to number of ws_
def test_numCC_ws_():
    assert len(kcca_ft.weights_) == 2
    
# Test that number of views is equal to number of comps_
def test_numCC_comps_():
    assert len(kcca_ft.components_) == 2
    
#Test that validate() runs
def test_validate():
    accuracy = kcca_ft.validate([test1, test2])
    assert (len(accuracy[0]) == 4 and len(accuracy[1]) == 5)
    
# Test that weights from fit equals fit.transform weights
def test_ktype_weights():
    assert kcca_t.weights_ == kcca_f
    
# Test that components from transform equals fit.transform weights
def test_ktype_components():
    assert np.allclose(kcca_ft.components_, kcca_t.components_)
    
# Test that gaussian kernel runs
def test_ktype_gaussian():
    kgauss = KCCA(ktype = 'gaussian', reg = 0.0001, n_components = 2, sigma=2.0)
    kgauss.fit_transform([train1, train2])
    assert len(kgauss.components_) == 2
    
# Test that polynomial kernel runs
def test_ktype_polynomial():
    kpoly = KCCA(ktype = 'poly', reg = 0.0001, n_components = 2, degree=3)
    kpoly.fit_transform([train1, train2])
    assert len(kpoly.components_) == 2

### Testing helper functions
np.random.seed(30)
a = np.random.uniform(0,1,(10,10))
b = np.random.uniform(0,1,(10,10))
c = np.ones((3,3))
c[1] = 0
    
# Test that _rowcorr works
def test_rowcorr():
    cs = _rowcorr(a,b)
    assert len(cs) == 10
    
# Test that _zscore works
def test_zscore(): 
    new = _zscore(c)
    assert (new[0] == new[2]).all()

# Test listdot works
def test_listdot(): 
    x = np.ones((10,10))
    y = 3*np.ones((10,10))
    ld = _listdot(x,y)
    assert ld == [30.0]*10
    
# Test _listcorr works
def test_listcorr():
    d = [a,b,a,b]
    lc = _listcorr(d)
    assert len(lc) == 10

# Test make_kernel
def test_make_kernel(): 
    lkernel = _make_kernel(c, normalize=True, ktype="linear", sigma=1.0, degree=2)
    gkernel = _make_kernel(c, normalize=True, ktype="gaussian", sigma=1.0, degree=2)
    pkernel = _make_kernel(c, normalize=True, ktype="poly", sigma=1.0, degree=2)
    assert lkernel.shape==gkernel.shape==pkernel.shape == (3,3)
    
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
def test_int_sigma():
    with pytest.raises(ValueError):
        kcca_f = KCCA(ktype ="gaussian", reg = 0.001, n_components = 1, sigma =1)

# Test error handling
def test_neg_degree():
    with pytest.raises(ValueError):
        kcca_g = KCCA(ktype ="poly", reg = 0.001, n_components = 1, degree =-1)

# Test error handling
def test_float_degree():
    with pytest.raises(ValueError):
        kcca_h = KCCA(ktype ="poly", reg = 0.001, n_components = 1, degree =1.0)

# Test error handling
def test_neg_cutoff():
    with pytest.raises(ValueError):
        kcca_i = KCCA(ktype ="poly", reg = 0.001, n_components = 1, cutoff= -1)

# Test error handling
def test_inf_cutoff():
    with pytest.raises(ValueError):
        kcca_j = KCCA(ktype ="poly", reg = 0.001, n_components = 1, cutoff= 1)
        
# Test if error when transform before fit
def test_no_weights():
    with pytest.raises(NameError):
        kcca_b = KCCA(ktype ="linear", reg = 0.001, n_components = 1)
        kcca_b.transform([train1, train2])
        
    