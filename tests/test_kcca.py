#unit tests

from multiview.embed.kcca import KCCA
import numpy as np

x = np.array([[1.,1.,3.],[2.,3.,2.],[1.,1.,1.],[1.,1.,2.],
              [2.,2.,3.],[3,3,2],[1,3,2],[4,3,5],[5,5,5]])
y = np.array([[4,4,-1.07846],[3,3,1.214359],[2,2,0.307180],
              [2,3,-0.385641],[2,1,-0.078461],[1,1,1.61436],
              [1,2,0.81436],[2,1,-0.06410],[1,2,1.54590]])

testcca = KCCA(kernelcca = False, reg = 0.0001, numCC = 2)
testcca.train([x, y])

print(testcca.comps_)

# Test that numCC is equal to number of cancorrs_
def test_numCC_cancorrs_():
    assert len(testcca.cancorrs_) == 2

# Test that numCC is equal to number of ws_
def test_numCC_ws_():
    assert len(testcca.ws_) == 2
    
# Test that numCC is equal to number of comps_
def test_numCC_comps_():
    assert len(testcca.comps_) == 2

# Test that numCC is equal to number of ev_
def test_numCC_ev_():
    assert len(testcca.compute_ev([x, y])) == 2
 
# Test that linear kernel works
def test_ktype_linear():
    klinear = KCCA(ktype = 'linear', reg = 0.0001, numCC = 2, gausigma=2)
    klinear.train([x, y])
    assert len(klinear.comps_) == 2
    
# Test that gaussian kernel works
def test_ktype_gaussian():
    kgauss = KCCA(ktype = 'gaussian', reg = 0.0001, numCC = 2, gausigma=2)
    kgauss.train([x, y])
    assert len(kgauss.comps_) == 2
    
# Test that polynomial kernel works
def test_ktype_polynomial():
    kpoly = KCCA(ktype = 'poly', reg = 0.0001, numCC = 2, degree=3)
    kpoly.train([x, y])
    assert len(kpoly.comps_) == 2


# Define two latent variables (number of samples x 1)
latvar1 = np.random.randn(1000,)
latvar2 = np.random.randn(1000,)

# Define independent components for each dataset (number of observations x dataset dimensions)
indep1 = np.random.randn(1000, 4)
indep2 = np.random.randn(1000, 5)

# Create two datasets, with each dimension composed as a sum of 75% one of the latent variables and 25% independent component
data1 = 0.25*indep1 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2)).T
data2 = 0.25*indep2 + 0.75*np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

# Split each dataset into two halves: training set and test set
train1 = data1[:1000//2]
train2 = data2[:1000//2]
test1 = data1[1000//2:]
test2 = data2[1000//2:]

# Test validate and prediction
def test_validate():
    cca = KCCA(kernelcca = False, reg = 0., numCC = 4)
    cca.train([train1, train2])
    testcorrs = cca.validate([test1, test2])
    assert len(testcorrs) == 2

    
    
    
    
    
    
    
    
    
    
    