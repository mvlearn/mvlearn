# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:24:42 2019

@author: arman
"""

import pytest
import numpy as np
from mvlearn.embed.mvmds import MVMDS
import math
from sklearn.metrics import euclidean_distances

'''
DATA INITIALIZATION
'''

@pytest.fixture(scope='module')
def data():
    
    N = 50
    D1 = 5
    D2 = 7
    D3 = 4
    
    np.random.seed(seed=5)
    first = np.random.rand(N,D1)
    second = np.random.rand(N,D2)
    third = np.random.rand(N,D3)
    random_views = [first, second, third]
    samp_views = [np.array([[1,4,0,6,2,3],
                        [2,5,7,1,4,3],
                        [9,8,5,4,5,6]]),             
                    np.array([[2,6,2,6],
                        [9,2,7,3],
                        [9,6,5,2]])]
    
    first_wrong = np.random.rand(N,D1)
    second_wrong = np.random.rand(N-1,D1)
    wrong_views = [first_wrong, second_wrong]
    
    dep_views = [np.array([[1,2,3],[1,2,3],[1,2,3]]),
                 np.array([[1,2,3],[1,2,3],[1,2,3]])]
    
    return {'wrong_views' : wrong_views, 'dep_views' : dep_views,
            'random_views' : random_views,
            'samp_views': samp_views}

'''
TESTS
'''


def test_component_num_greater(data):
    mvmds = MVMDS(n_components = len(data['random_views'][0] + 1))
    comp = mvmds.fit_transform(data['random_views'])
    
    assert len(comp) == len(data['random_views'][0])       

          
def test_fit_transform_values(data):
    n_components = len(data['samp_views'][0])
    mvmds = MVMDS(n_components = n_components)
    comp = mvmds.fit_transform(data['samp_views'])
    comp2 = np.array([[-0.81330129,  0.07216426,  0.5773503],
                      [0.34415456, -0.74042171,  0.5773503],
                      [0.46914673,  0.66825745, 0.5773503]])
    
    # Last component calculation varies across Python implementations.
    np.testing.assert_almost_equal(
        np.abs(comp[:,n_components-1]),
        np.abs(comp2[:,n_components-1])
    )
            
def test_fit_transformdifferent_wrong_samples(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS()
        mvmds.fit_transform(data['wrong_views'])

#This is about taking in views that are the same.

def test_depend_views(data):
    mvmds = MVMDS()
    fit = mvmds.fit_transform(data['dep_views'])
    
    for i in range(fit.shape[0]):
        for j in range(fit.shape[1]):     
            assert math.isnan(fit[i,j])

'''
Parameter Checks
'''

def test_fit_transform_values_0():
    with pytest.raises(ValueError):
        MVMDS(n_components=0)
        
def test_fit_transform_values_neg():
    with pytest.raises(ValueError):
        MVMDS(n_components=-4)

def test_num_iter_value_fail():
    with pytest.raises(ValueError):
        MVMDS(num_iter=0)

def test_dissimilarity_wrong():
    with pytest.raises(ValueError):
        MVMDS(dissimilarity=3)

def test_dissimilarity_euclidean():
    with pytest.raises(ValueError):
        MVMDS(n_components=-3)

def test_dissimilarity_precomputed_euclidean(data):
    test_views = []
    for i in data['samp_views']:
        test_views.append(euclidean_distances(i))
    mvmds1 = MVMDS(dissimilarity='euclidean')
    mvmds2 = MVMDS(dissimilarity='precomputed')

    fit1 = mvmds1.fit_transform(data['samp_views'])
    fit2 = mvmds2.fit_transform(test_views)

    np.testing.assert_almost_equal(np.abs(fit2), np.abs(fit1))
