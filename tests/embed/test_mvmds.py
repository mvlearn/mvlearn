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
    mvmds = MVMDS(len(data['random_views'][0] + 1))
    comp = mvmds.fit_transform(data['random_views'])
    
    assert len(comp) == len(data['random_views'][0])       

          
def test_fit_transform_values(data):
    mvmds = MVMDS(len(data['samp_views'][0]))
    comp = mvmds.fit_transform(data['samp_views'])
    comp2 = np.array([[-0.81330129,  0.07216426,  0.57735027],
           [ 0.34415456, -0.74042171,  0.57735027],
           [ 0.46914673,  0.66825745,  0.57735027]])
    
    for i in range(comp.shape[0]):
        for j in range(comp.shape[1]):
            assert comp[i,j]-comp2[i,j] < .000001
            
def test_fit_transformdifferent_wrong_samples(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(2)
        mvmds.fit_transform(data['wrong_views'])

#This is about taking in views that are the same.

def test_depend_views(data):
    mvmds = MVMDS(2)
    fit = mvmds.fit_transform(data['dep_views'])
    
    for i in range(fit.shape[0]):
        for j in range(fit.shape[1]):     
            assert math.isnan(fit[i,j])

'''
Parameter Checks
'''

def test_fit_transform_values_0(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(n_components=0)
        mvmds.fit_transform(data['samp_views'])

        
def test_fit_transform_values_neg(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(n_components=-4)
        comp = mvmds.fit_transform(data['samp_views'])

def check_num_iter(data):
    with pytest.raises(ValueError):
        
        mvmds = MVMDS(n_components=-3)
        mvmds.fit_transform(data['samp_views'])


def check_dist_wrong(data):
    with pytest.raises(ValueError):

        mvmds = MVMDS(n_components=-3,distance=3)
        mvmds.fit_transform(data['samp_views'])        

def check_dist_true(data):
    with pytest.raises(ValueError):
        
        mvmds = MVMDS(n_components=-3,distance=True)
        mvmds.fit_transform(data['samp_views'])

def check_dist_true_vals(data):
    test_views = []
    for i in data['samp_views']:
        test_views.append(euclidean_distances(i))
    mvmds1 = MVMDS(n_components=2,distance=False)
    mvmds2 = MVMDS(n_components=2,distance=True)

    fit1 = mvmds1.fit_transform(data['samp_views'])
    fit2 = mvmds2.fit_transform(test_views)

    for i in range(fit1.shape[0]):
        for j in range(fit1.shape[1]):
            assert fit1[i,j]-fit2[i,j] < .000001