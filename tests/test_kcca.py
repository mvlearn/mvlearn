# KCCA Unit Tests

from multiview.embed.kcca import KCCA, _zscore
import numpy as np

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
    kgauss = KCCA(ktype = 'gaussian', reg = 0.0001, n_components = 2, sigma=2)
    kgauss.fit_transform([train1, train2])
    assert len(kgauss.components_) == 2
    
# Test that polynomial kernel runs
def test_ktype_polynomial():
    kpoly = KCCA(ktype = 'poly', reg = 0.0001, n_components = 2, degree=3)
    kpoly.fit_transform([train1, train2])
    assert len(kpoly.components_) == 2
    
    
    
    
    
    
    
    
    
    
    