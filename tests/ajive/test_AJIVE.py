import unittest

import numpy as np
import pandas as pd
import pytest
from mvlearn.ajive.ajive import ajive
from scipy.sparse import csr_matrix

class TestFig2Runs(unittest.TestCase):

    @classmethod
    def setUp(self):
        
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
        
        X = V1_indiv_tot + V1_joint + V1_noise
        
        Y = V2_indiv + V2_joint + V2_noise

        
        obs_names = ['sample_{}'.format(i) for i in range(X.shape[0])]
        var_names = {'x': ['x_var_{}'.format(i) for i in range(X.shape[1])],
                     'y': ['y_var_{}'.format(i) for i in range(Y.shape[1])]}

        X = pd.DataFrame(X, index=obs_names, columns=var_names['x'])
        Y = pd.DataFrame(Y, index=obs_names, columns=var_names['y'])

        jive = ajive(init_signal_ranks={'x': 2, 'y': 3}).fit(blocks={'x': X,\
                    'y': Y})

        self.ajive = jive
        self.X = X
        self.Y = Y
        self.obs_names = obs_names
        self.var_names = var_names

    def test_has_attributes(self):
        """
        Check AJIVE has important attributes
        """
        self.assertTrue(hasattr(self.ajive, 'blocks'))
        self.assertTrue(hasattr(self.ajive, 'common'))
        self.assertTrue(hasattr(self.ajive.blocks['x'], 'joint'))
        self.assertTrue(hasattr(self.ajive.blocks['x'], 'individual'))
        self.assertTrue(hasattr(self.ajive.blocks['y'], 'joint'))
        self.assertTrue(hasattr(self.ajive.blocks['y'], 'individual'))

    def test_correct_estimates(self):
        """
        Check AJIVE found correct rank estimates
        """
        self.assertEqual(self.ajive.common.rank, 1)
        self.assertEqual(self.ajive.blocks['x'].individual.rank, 1)
        self.assertEqual(self.ajive.blocks['y'].individual.rank, 3)

    def test_matrix_decomposition(self):
        """
        check X_centered = I + J + E
        """
        X_cent = self.X - self.X.mean(axis=0)
        Rx = X_cent - (self.ajive.blocks['x'].joint.full_ +
                       self.ajive.blocks['x'].individual.full_ +
                       self.ajive.blocks['x'].noise_)

        self.assertTrue(np.allclose(Rx, 0))

        Y_cent = self.Y - self.Y.mean(axis=0)
        Ry = Y_cent - (self.ajive.blocks['y'].joint.full_ +
                       self.ajive.blocks['y'].individual.full_ +
                       self.ajive.blocks['y'].noise_)

        self.assertTrue(np.allclose(Ry, 0))

    def test_common_SVD(self):
        """
        Check common SVD
        """
        U, D, V = self.ajive.common.get_UDV()
        rank = self.ajive.common.rank
        n = self.X.shape[0]
        d = sum(self.ajive.init_signal_ranks.values())
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

    def test_block_specific_SVDs(self):
        """
        Check each block specific SVD
        """
        U, D, V = self.ajive.blocks['x'].joint.get_UDV()
        rank = 1
        n, d = self.X.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

        U, D, V = self.ajive.blocks['x'].individual.get_UDV()
        rank = 1
        n, d = self.X.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

        U, D, V = self.ajive.blocks['y'].joint.get_UDV()
        rank = 1
        n, d = self.Y.shape
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

    def test_names(self):
        self.assertEqual(set(self.ajive.common.obs_names()),
                         set(self.obs_names))
        self.assertEqual(set(self.ajive.common.scores_.index),
                         set(self.obs_names))

        self.assertEqual(set(self.ajive.blocks['x'].joint.obs_names()),
                         set(self.obs_names))

        self.assertEqual(set(self.ajive.blocks['x'].joint.scores_.index),
                         set(self.obs_names))

        self.assertEqual(set(self.ajive.blocks['x'].joint.var_names()),
                         set(self.var_names['x']))

        self.assertEqual(set(self.ajive.blocks['x'].individual.obs_names()),
                         set(self.obs_names))

        self.assertEqual(set(self.ajive.blocks['x'].individual.var_names()),
                         set(self.var_names['x']))

    def test_parallel_runs(self):
        """
        Check wedin/random samples works with parallel processing.
        """
        jive = ajive(init_signal_ranks={'x': 2, 'y': 3}, n_jobs=-1)
        jive.fit(blocks={'x': self.X, 'y': self.Y})
        self.assertTrue(hasattr(jive, 'blocks'))

    def test_list_input(self):
        """
        Check AJIVE can take a list input.
        """
        jive = ajive(init_signal_ranks=[2, 3])
        jive.fit(blocks=[self.X, self.Y])
        self.assertTrue(set(jive.block_names) == set([0, 1]))

    def test_dont_store_full(self):
        """
        Make sure setting store_full = False works
        """
        jive = ajive(init_signal_ranks=[2, 3], store_full=False)
        jive.fit(blocks=[self.X, self.Y])

        self.assertTrue(jive.blocks[0].joint.full_ is None)
        self.assertTrue(jive.blocks[0].individual.full_ is None)
        self.assertTrue(jive.blocks[1].joint.full_ is None)
        self.assertTrue(jive.blocks[1].individual.full_ is None)

    def test_rank0(self):
        """
        Check setting joint/individual rank to zero works
        """
        jive = ajive(init_signal_ranks=[2, 3], joint_rank=0)
        jive.fit(blocks=[self.X, self.Y])
        self.assertTrue(jive.common.rank == 0)
        self.assertTrue(jive.blocks[0].joint.rank == 0)
        self.assertTrue(jive.blocks[0].joint.scores_ is None)

        jive = ajive(init_signal_ranks=[2, 3], indiv_ranks=[0, 1])
        jive.fit(blocks=[self.X, self.Y])
        self.assertTrue(jive.blocks[0].individual.rank == 0)
        self.assertTrue(jive.blocks[0].individual.scores_ is None)

    def test_centering(self):
        xmean = self.X.mean(axis=0)
        ymean = self.Y.mean(axis=0)

        self.assertTrue(np.allclose(self.ajive.centers_['x'], xmean))
        self.assertTrue(np.allclose(self.ajive.blocks['x'].joint.m_, xmean))
        self.assertTrue(np.allclose(self.ajive.blocks['x'].individual.m_, \
                                    xmean))

        self.assertTrue(np.allclose(self.ajive.centers_['y'], ymean))
        self.assertTrue(np.allclose(self.ajive.blocks['y'].joint.m_, ymean))
        self.assertTrue(np.allclose(self.ajive.blocks['y'].individual.m_, \
                                    ymean))

        # no centering
        jive = ajive(init_signal_ranks={'x': 2, 'y': 3}, center=False)
        jive = jive.fit(blocks={'x': self.X, 'y': self.Y})
        self.assertTrue(jive.centers_['x'] is None)
        self.assertTrue(jive.centers_['y'] is None)

        # only center x
        jive = ajive(init_signal_ranks={'x': 2, 'y': 3}, center={'x': True, \
                      'y': False})
        jive = jive.fit(blocks={'x': self.X, 'y': self.Y})
        self.assertTrue(np.allclose(jive.centers_['x'], xmean))
        self.assertTrue(jive.centers_['y'] is None)
        
if __name__ == '__main__':
    unittest.main()


def svd_checker(U, D, V, n, d, rank):
    checks = {}

    # scores shape
    checks['scores_shape'] = U.shape == (n, rank)

    # scores have orthonormal columns
    checks['scores_ortho'] = np.allclose(np.dot(U.T, U), np.eye(rank))

    # singular values shape
    checks['svals_shape'] = D.shape == (rank, )

    # singular values are in non-increasing order
    svals_nonincreasing = True
    for i in range(len(D) - 1):
        if D[i] < D[i+1]:
            svals_nonincreasing = False
    checks['svals_nonincreasing'] = svals_nonincreasing

    # loadings shape
    checks['loading_shape'] = V.shape == (d, rank)

    # loadings have orthonormal columns
    checks['loadings_ortho'] = np.allclose(np.dot(V.T, V), np.eye(rank))
    
    return checks


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
    jive = ajive(init_signal_ranks= [2,2])
    jive.fit(blocks = dat)
    blocks = jive.predict()
    assert blocks[0]['joint'].shape == blocks[0]['individual'].shape        

def test_joint_noise_length(data):
    dat = data['same_views']
    jive = ajive(init_signal_ranks= [2,2])
    jive.fit(blocks = dat)
    blocks = jive.predict()
    assert blocks[0]['joint'].shape == blocks[0]['noise'].shape        

          
def test_joint(data):
    dat = data['same_views']
    jive = ajive(init_signal_ranks= [2,2])
    jive.fit(blocks = dat)
    blocks = jive.predict()
    for i in np.arange(100):
        j = np.sum(blocks[0]['joint'][i] == blocks[1]['joint'][i])
        assert j == 20

def test_indiv(data):
    dat = data['same_views']
    jive = ajive(init_signal_ranks= [2,2])
    jive.fit(blocks = dat)
    blocks = jive.predict()
    for i in np.arange(100):
        j = np.sum(blocks[0]['individual'][i] == blocks[1]['individual'][i])
        assert j == 20

#Sees whether incorrect signals will work
def test_wrong_sig(data):
    dat = data['diff_views']
    jive = ajive(init_signal_ranks= [-1,-4])
    try:
        jive.fit(blocks=dat)
        j = 0
    except:
        j = 1
    assert j == 1

def test_check_sparse(data):
    dat = data['sparse_views']
    spar_mat = dat[0]
    assert np.sum(spar_mat == 0) > np.sum(spar_mat != 0)
    jive = ajive(init_signal_ranks= [2,2])
    jive.fit(blocks = dat)
    blocks = jive.predict()
    assert np.sum(np.sum(blocks[0]['individual'] == 0)) > \
    np.sum(np.sum(blocks[0]['individual'] != 0)) 

#Check valueerror for general linear operators
def test_check_gen_lin_op_scipy(data):
    with pytest.raises(TypeError):
        dat = data['bad_views']
        jive = ajive(init_signal_ranks= [2,2])
        jive.fit(blocks = dat)

def test_check_joint_rank_large(data):
    with pytest.raises(ValueError):
        dat = data['same_views']
        jive = ajive(init_signal_ranks= [2,2], joint_rank=5)
        jive.fit(blocks = dat)

def test_decomp_not_computed_ranks():
    with pytest.raises(ValueError):
        jive = ajive(init_signal_ranks=[2,2])
        jive.get_ranks()

def test_indiv_rank(data):
    dat = data['same_views']
    jive = ajive(init_signal_ranks= [2,2], indiv_ranks=[2,1])
    jive.fit(blocks = dat)
    assert jive.indiv_ranks[0] == 2

def test_joint_rank(data):
    dat = data['same_views']
    jive = ajive(init_signal_ranks= [2,2], joint_rank=2)
    jive.fit(blocks = dat)
    assert jive.joint_rank == 2

def test_is_fit():
    jive = ajive(init_signal_ranks = [2,2],joint_rank=2)
    assert jive.is_fit == False

def test_n_randdir():
    jive = ajive(init_signal_ranks = [2,2],n_randdir_samples=5)
    assert jive.n_randdir_samples == 5

def test_n_jobs():
    jive = ajive(init_signal_ranks = [2,2], n_jobs=4)
    assert jive.n_jobs == 4

def test_n_wedin():
    jive = ajive(init_signal_ranks = [2,2], n_wedin_samples = 6)
    assert jive.n_wedin_samples == 6

#Plotting

def test_plot_diag(data):
    x = data['same_views']
    ajive.data_block_heatmaps(x)
    p = 1
    assert p == 1

def test_ajive_plot(data):
    x = data['same_views']
    ajive.ajive_full_estimate_heatmaps(blocks=x)
    p = 1
    assert p == 1
    

    