import unittest

import numpy as np
import pandas as pd
from mvlearn.factorization.ajive_utils.pca import pca

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
        X = self.X
        svals = self.pca.svals_
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

