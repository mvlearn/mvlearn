# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:55:10 2020

@author: arman
"""


import pytest
import numpy as np
from mvlearn.embed.mvmds import MVMDS
from mvlearn.jive.AJIVE import AJIVE
import math

'''
DATA INITIALIZATION
'''

@pytest.fixture(scope='module')
def data():
    
    np.random.seed(12)

    # First View
    V1_joint = np.bmat([[-1 * np.ones((10, 20))],
                           [np.ones((10, 20))]])
    
    V1_joint = np.bmat([np.zeros((20, 80)), V1_joint])
    
    V1_indiv_t = np.bmat([[np.ones((4, 50))],
                            [-1 * np.ones((4, 50))],
                            [np.zeros((4, 50))],
                            [np.ones((4, 50))],
                            [-1 * np.ones((4, 50))]])
    
    V1_indiv_b = np.bmat([[np.ones((5, 50))],
                            [-1 * np.ones((10, 50))],
                            [np.ones((5, 50))]])
    
    V1_indiv_tot = np.bmat([V1_indiv_t, V1_indiv_t])
    
    V1_noise = np.random.normal(loc=0, scale=1, size=(20, 100))
    
    
    # Second View
    V2_joint = np.bmat([[np.ones((10, 10))],
                          [-1*np.ones((10, 10))]])
    
    V2_joint = 5000 * np.bmat([V2_joint, np.zeros((20, 10))])
    
    V2_indiv = 5000 * np.bmat([[-1 * np.ones((5, 20))],
                                  [np.ones((5, 20))],
                                  [-1 * np.ones((5, 20))],
                                  [np.ones((5, 20))]])
    
    V2_noise = 5000 * np.random.normal(loc=0, scale=1, size=(20, 20))
    
    # View Construction
    
    V1 = V1_indiv_tot + V1_joint + V1_noise
    
    V2 = V2_indiv + V2_joint + V2_noise

    Views_Same = [V1, V1]
    Views_Different = [V1, V2]
    
    return {'same_views' : Views_Same, 'diff_views' : Views_Different}

'''
TESTS
'''


def test_component_num_greater(data):
    mvmds = MVMDS(len(data['random_views'][0] + 1))
    comp = mvmds.fit_transform(data['random_views'])
    
    assert len(comp) == len(data['random_views'][0])       

          
def test_joint(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2])
    ajive.fit(blocks = dat)
    blocks = ajive.get_full_block_estimates()
    for i in np.arange(100):
        j = np.sum(blocks[0]['joint'][i] == blocks[1]['joint'][i])
        assert j == 20

def test_indiv(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2])
    ajive.fit(blocks = dat)
    blocks = ajive.get_full_block_estimates()
    for i in np.arange(100):
        j = np.sum(blocks[0]['individual'][i] == blocks[1]['individual'][i])
        assert j == 20

#Sees whether incorrect signals will work
def test_wrong_sig(data):
    dat = data['diff_views']
    ajive = AJIVE(init_signal_ranks= [-1,-4])
    try:
        ajive.fit(blocks=dat)
        j = 0
    except:
        j = 1
    assert j == 1
        
def test_fit_transformdifferent_wrong_samples(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(2)
        comp = mvmds.fit_transform(data['wrong_views'])

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
        comp = mvmds.fit_transform(data['samp_views'])

        
def test_fit_transform_values_neg(data):
    with pytest.raises(ValueError):
       
        mvmds = MVMDS(n_components=-4)
        comp = mvmds.fit_transform(data['samp_views'])

def check_num_iter(data):
    with pytest.raises(ValueError):
        
        mvmds = MVMDS(n_components=-3)
        comp = mvmds.fit_transform(data['samp_views'])