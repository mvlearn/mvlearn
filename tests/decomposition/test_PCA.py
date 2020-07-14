import unittest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from scipy.sparse import csr_matrix
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from mvlearn.decomposition.ajive_utils.pca import pca, _arg_checker, _default_obs_names, svd2pd, _unnorm_scores, pca_reconstruct, _safe_frob_norm


class TestPCA(unittest.TestCase):

    @classmethod
    def setUp(self):
        n = 100
        d = 20
        n_components = 10
        obs_names = ['sample_{}'.format(i) for i in range(n)]
        var_names = ['var_{}'.format(i) for i in range(d)]

        X = pd.DataFrame(np.random.normal(size=(n, d)),
                         index=obs_names, columns=var_names)
        X_cent = X - X.mean(axis=0)

        PCA = pca(n_components=n_components).fit(X)

        # store these for testing
        self.n = n
        self.d = d
        self.n_components = n_components
        self.obs_names = obs_names
        self.var_names = var_names
        self.X = X
        self.X_cent = X_cent
        self.pca = PCA

    def test_has_attributes(self):
        """
        Check AJIVE has important attributes
        """
        self.assertTrue(hasattr(self.pca, 'scores_'))
        self.assertTrue(hasattr(self.pca, 'svals_'))
        self.assertTrue(hasattr(self.pca, 'loadings_'))
        self.assertTrue(hasattr(self.pca, 'm_'))
        self.assertTrue(hasattr(self.pca, 'frob_norm_'))

    def test_name_extraction(self):
        self.assertTrue(set(self.pca.obs_names()), set(self.obs_names))
        self.assertTrue(set(self.pca.var_names()), set(self.var_names))

    def test_shapes(self):
        self.assertEqual(self.pca.shape_, (self.n, self.d))

        self.assertEqual(self.pca.scores_.shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores().shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores(norm=True).shape, (self.n,\
                         self.n_components))
        self.assertEqual(self.pca.scores(norm=False).shape, (self.n,\
                         self.n_components))

        self.assertEqual(self.pca.loadings_.shape, (self.d, self.n_components))
        self.assertEqual(self.pca.loadings().shape, (self.d,\
                         self.n_components))

        self.assertEqual(self.pca.svals_.shape, (self.n_components, ))
        self.assertEqual(self.pca.svals().shape, (self.n_components, ))

    def test_SVD(self):
        """
        Check the SVD decomposition is correct.
        """
        U, D, V = self.pca.get_UDV()
        n, d = self.X.shape
        rank = self.n_components
        checks = svd_checker(U, D, V, n, d, rank)
        self.assertTrue(all(checks.values()))

    def test_reconstruction(self):
        """
        We can reconstruct the original data matrix exactly from the full
        reconstruction.
        """
        PCA= pca().fit(self.X)
        self.assertTrue(np.allclose(self.X, PCA.predict_reconstruction()))

    def test_frob_norm(self):
        """
        Check Frobenius norm is calculated correctly whether the full
        or partial PCA is computed.
        """
        true_frob_norm = np.linalg.norm(self.X_cent, ord='fro')
        PCA = pca(n_components=None).fit(self.X)
        self.assertTrue(np.allclose(PCA.frob_norm_, true_frob_norm))

        # TODO: this is failing, it could be a numerical issue.
        PCA = pca(n_components=3).fit(self.X)
        self.assertTrue(np.allclose(PCA.frob_norm_, true_frob_norm))

    def test_centering(self):
        """
        Make sure PCA computes the correct centers. Also check center=False
        works correctly.
        """
        self.assertTrue(np.allclose(self.pca.m_, self.X.mean(axis=0)))

        # no centering
        PCA = pca(n_components=4, center=False).fit(self.X)
        self.assertTrue(PCA.m_ is None)

        Z = np.random.normal(size=(20, self.X.shape[1]))
        V = PCA.loadings_.values
        self.assertTrue(np.allclose(PCA.predict_scores(Z), np.dot(Z, V)))

    def test_projection(self):
        """
        Make sure projection onto loadings subspace works
        """
        Z = np.random.normal(size=(10, self.d))

        m = self.X.values.mean(axis=0)
        V = self.pca.loadings_.values

        Z_cent = Z - m.reshape(-1)
        A = np.dot(Z_cent, V)
        B = self.pca.predict_scores(Z)
        
        self.assertTrue(np.allclose(A, B))
    
    def test_functs(self):
        """
        Make sure other functions work
        """
        c = []
        c.append(not isinstance(self.pca.svals(np=False), np.ndarray))
        c.append(isinstance(self.pca.svals(np=True), np.ndarray))
        
        c.append(isinstance(self.pca.scores(norm=True, np=True), np.ndarray))
        c.append(not isinstance(self.pca.scores(norm=True, np=False),
                                np.ndarray))
        
        c.append(isinstance(self.pca.loadings(np=True),np.ndarray))
        c.append(not isinstance(self.pca.loadings(np=False),np.ndarray))
        
        c.append(isinstance(self.pca.comp_names(),np.ndarray))
        
        j = self.pca
        j.set_comp_names(self.pca.comp_names())
        
        c.append(isinstance(self.pca.obs_names(),np.ndarray))

        
        self.assertTrue(all(c))

    def test_get_params(self):
        """
        Make sure get_params works for pca class.
        """
        d = self.pca.get_params()
        self.assertEqual(d['n_components'], self.n_components)
        self.assertEqual(d['center'], 'mean')

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

def test_dunder_repr():
    """
    Make sure __repr__ function works.
    """ 
    pca1 = pca(n_components=2, center='mean')
    compare = 'pca object, nothing has been computed yet'
    assert_equal(pca1.__repr__(), compare)
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    pca1.fit(X)
    compare = 'Rank {} pca of a {} matrix'.format(2, X.shape)
    assert_equal(pca1.__repr__(), compare)

def test_n_components_None():
    pca1 = pca()
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    pca1.fit(X)
    d = pca1.get_params()
    assert_equal(d['n_components'], X.shape[1])

def test_from_precomputed():
    pca1 = pca.from_precomputed(pca)
    assert pca1.shape_[0] == None
    assert pca1.shape_[1] == None

    scores = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    pca2 = pca.from_precomputed(pca, scores=scores)
    df = pd.DataFrame(scores, columns=['comp_0', 'comp_1', 'comp_2'])
    assert_frame_equal(df, pca2.scores_)

    svals = np.ones((3,))
    pca3 = pca.from_precomputed(pca, scores=scores, svals=svals)
    series = pd.Series(svals, index=['comp_0', 'comp_1', 'comp_2'])
    assert_series_equal(series, pca3.svals_)

    pca4 = pca.from_precomputed(pca, var_expl_prop=5)
    assert pca4.var_expl_prop_ == 5

def test_scores():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    pca2 = pca()
    pca2.fit(X)
    scores = np.array([[-7.794229,  0.      , -0.      ],
                       [-2.598076,  0.      ,  0.      ],
                       [ 2.598076,  0.      ,  0.      ],
                       [ 7.794229, -0.      ,  0.      ]])
    assert_almost_equal(scores, pca2.scores(norm=False, np=True), decimal=4)

def test_predict_reconstruction():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    pca2 = pca()
    pca2.fit(X)
    recon = pca2.predict_reconstruction(X)
    R = np.array([[-47.7852752, -46.7852752, -45.7852752],
                  [-12.9284251, -11.9284251, -10.9284251],
                  [ 21.9284251,  22.9284251,  23.9284251],
                  [ 56.7852752,  57.7852752,  58.7852752]])
    assert_almost_equal(R, recon, decimal=4)

def test_arg_checker():
    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    out = _arg_checker(X, n_components=2)
    shape, obs_names, var_names, n_components = out
    assert shape == (X.shape)
    assert obs_names is None
    assert var_names is None
    assert n_components == 2

def test_default_obs_names():
    out = _default_obs_names(4)
    assert out == [0, 1, 2, 3]

def test_unnorm_scores():
    U = np.ones((4,))
    D = np.ones((4,))
    UD = U.reshape(1, -1) * np.array(D)
    assert_equal(UD, _unnorm_scores(U, D))

    U = np.ones((4,1))
    D = np.ones(4,)
    UD = U.reshape(1, -1) * np.array(D)
    assert_equal(UD, _unnorm_scores(U, D))


def test_pca_reconstruct():
    U = np.ones((4,))
    D = np.ones((4,))
    V = np.ones((2,4))
    m = np.ones((2,1)).reshape(1,-1)
    UD = _unnorm_scores(U, D)
    R = (np.dot(UD, V.T) + m).squeeze()
    recon = pca_reconstruct(U, D, V, m=m)
    assert_equal(R, recon)

def test_safe_frob_norm():
    X = csr_matrix([[0, 1], [1, 2]])
    norm = np.sqrt(sum(X.data ** 2))
    assert_equal(norm, _safe_frob_norm(X))


