# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:55:10 2020

@author: arman
"""


import pytest
import numpy as np
from mvlearn.jive.AJIVE import AJIVE
from scipy.sparse import csr_matrix


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
    
    V1_indiv_tot = np.bmat([V1_indiv_t, V1_indiv_b])
    
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

    # Creating Sparse views
    V1_sparse = np.array(np.zeros_like(V1))
    V2_sparse = np.array(np.zeros_like(V2))
    V1_sparse[0,0] = 1
    V2_sparse[0,0] = 3
    V1_Bad = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    V2_Bad = csr_matrix([[1, 2, 3], [7, 0, 3], [1, 2, 2]])
   
    Views_Same = [V1, V1]
    Views_Different = [V1, V2]
    Views_Sparse = [V1_sparse, V2_sparse]
    Views_Bad = [V1_Bad, V2_Bad]

    
    return {'same_views' : Views_Same, 'diff_views' : Views_Different,
            'sparse_views' : Views_Sparse, 'bad_views': Views_Bad}

'''
TESTS
'''


def test_joint_indiv_length(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2])
    ajive.fit(blocks = dat)
    blocks = ajive.get_full_block_estimates()
    assert blocks[0]['joint'].shape == blocks[0]['individual'].shape        

def test_joint_noise_length(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2])
    ajive.fit(blocks = dat)
    blocks = ajive.get_full_block_estimates()
    assert blocks[0]['joint'].shape == blocks[0]['noise'].shape        

          
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

def check_sparse(data):
    dat = data['sparse_views']
    spar_mat = dat[0]
    assert np.sum(spar_mat == 0) > np.sum(spar_mat != 0)
    ajive = AJIVE(init_signal_ranks= [2,2])
    ajive.fit(blocks = dat)
    blocks = ajive.get_full_block_estimates()
    assert np.sum(np.sum(blocks[0]['individual'] == 0)) > \
    np.sum(np.sum(blocks[0]['individual'] != 0)) 

#Check valueerror for general linear operators
def check_gen_lin_op_scipy(data):
    with pytest.raises(ValueError):
        dat = data['bad_views']
        ajive = AJIVE(init_signal_ranks= [2,2])
        ajive.fit(blocks = dat)

def check_joint_rank_large(data):
    with pytest.raises(ValueError):
        dat = data['same_views']
        ajive = AJIVE(init_signal_ranks= [2,2], joint_rank=5)
        ajive.fit(blocks = dat)

def decomp_not_computed_ranks():
    with pytest.raises(ValueError):
        ajive = AJIVE(init_signal_ranks=[2,2])
        ajive.get_ranks()

def test_indiv_rank(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2], indiv_ranks=[2,1])
    ajive.fit(blocks = dat)
    assert ajive.indiv_ranks[0] == 2

def test_joint_rank(data):
    dat = data['same_views']
    ajive = AJIVE(init_signal_ranks= [2,2], joint_rank=2)
    ajive.fit(blocks = dat)
    assert ajive.joint_rank == 2

def test_is_fit():
    ajive = AJIVE(init_signal_ranks = [2,2],joint_rank=2)
    assert ajive.is_fit == False

def test_n_randdir():
    ajive = AJIVE(init_signal_ranks = [2,2],n_randdir_samples=5)
    assert ajive.n_randdir_samples == 5

def test_n_jobs():
    ajive = AJIVE(init_signal_ranks = [2,2], n_jobs=4)
    assert ajive.n_jobs == 4

def test_n_wedin():
    ajive = AJIVE(init_signal_ranks = [2,2], n_wedin_samples = 6)
    assert ajive.n_wedin_samples == 6
