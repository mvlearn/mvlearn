# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:24:42 2019

@author: arman
"""

import pytest
import numpy as np
from multiview.embed.mvmds import MVMDS
import math

'''
DATA INITIALIZATION
'''

@pytest.fixture(scope='module')
def data():
    
    N = 50
    D1 = 5
    D2 = 7
    D3 = 4

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
    comp = mvmds.fit(data['random_views'])
    
    assert len(comp) == len(data['random_views'][0])       

       
def test_fit_values(data):
    mvmds = MVMDS(len(data['samp_views'][0]))
    comp = mvmds.fit(data['samp_views'])
    comp2 = np.array([[-0.81330129,  0.07216426,  0.57735027],
           [ 0.34415456, -0.74042171,  0.57735027],
           [ 0.46914673,  0.66825745,  0.57735027]])
    
    for i in range(comp.shape[0]):
        for j in range(comp.shape[1]):
            assert comp[i,j]-comp2[i,j] < .000001

    
def test_fit_transform_values(data):
    mvmds = MVMDS(len(data['samp_views'][0]))
    comp = mvmds.fit_transform(data['samp_views'])
    comp2 = np.array([[-0.81330129,  0.07216426,  0.57735027],
           [ 0.34415456, -0.74042171,  0.57735027],
           [ 0.46914673,  0.66825745,  0.57735027]])
    
    for i in range(comp.shape[0]):
        for j in range(comp.shape[1]):
            assert comp[i,j]-comp2[i,j] < .000001


def test_transform(data):
    mvmds = MVMDS(len(data['random_views'][0]))
    comp = mvmds.transform(data['random_views'])
    
    for i in range(len(comp)):
        for j in range(comp[i].shape[0]):     
            for k in range(comp[i].shape[1]):
                
                assert abs(comp[i][j,k] - \
                           data['random_views'][i][j,k]) < .000001


def test_fit_values_0(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(0)
        comp = mvmds.fit(data['samp_views'])

        
def test_fit_values_neg(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(-4)
        comp = mvmds.fit(data['samp_views'])

            
def test_fit_different_wrong_samples(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(2)
        comp = mvmds.fit(data['wrong_views'])

def test_fit_fit_transform_same(data):
    mvmds = MVMDS(2)
    comp_fit = mvmds.fit(data['samp_views'])
    comp_fit_transform = mvmds.fit_transform(data['samp_views'])
    
    for i in range(comp_fit.shape[0]):
        for j in range(comp_fit.shape[1]):
            assert comp_fit[i,j] - \
            comp_fit_transform[i,j] < .0000001


#This is about taking in views that are the same.

def test_depend_views(data):
    mvmds = MVMDS(2)
    fit = mvmds.fit(data['dep_views'])
    
    for i in range(fit.shape[0]):
        for j in range(fit.shape[1]):     
            assert math.isnan(fit[i,j])
