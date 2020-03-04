import unittest

import numpy as np
import pandas as pd
from jive.PCA import PCA
from jive.tests.utils import svd_checker


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

        pca = PCA(n_components=n_components).fit(X)

        # store these for testing
        self.n = n
        self.d = d
        self.n_components = n_components
        self.obs_names = obs_names
        self.var_names = var_names
        self.X = X
        self.X_cent = X_cent
        self.pca = pca

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
        self.assertEqual(self.pca.scores(norm=True).shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores(norm=False).shape, (self.n, self.n_components))

        self.assertEqual(self.pca.loadings_.shape, (self.d, self.n_components))
        self.assertEqual(self.pca.loadings().shape, (self.d, self.n_components))

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
        pca = PCA().fit(self.X)
        self.assertTrue(np.allclose(self.X, pca.predict_reconstruction()))

    def test_plots(self):
        """
        Check all plotting functions run without error
        """
        self.pca.plot_loading(comp=0)
        self.pca.plot_scores_hist(comp=1)
        self.pca.plot_scree()
        self.pca.plot_var_expl_prop()
        self.pca.plot_var_expl_cum()
        self.pca.plot_scores()
        self.pca.plot_scores_vs(comp=1, y=np.random.normal(size=self.X.shape[0]))
        # self.pca.plot_interactive_scores_slice(1, 3)

    def test_frob_norm(self):
        """
        Check Frobenius norm is calculated correctly whether the full
        or partial PCA is computed.
        """
        true_frob_norm = np.linalg.norm(self.X_cent, ord='fro')
        pca = PCA(n_components=None).fit(self.X)
        self.assertTrue(np.allclose(pca.frob_norm_, true_frob_norm))

        # TODO: this is failing, it could be a numerical issue.
        pca = PCA(n_components=3).fit(self.X)
        self.assertTrue(np.allclose(pca.frob_norm_, true_frob_norm))

    def test_centering(self):
        """
        Make sure PCA computes the correct centers. Also check center=False
        works correctly.
        """
        self.assertTrue(np.allclose(self.pca.m_, self.X.mean(axis=0)))

        # no centering
        pca = PCA(n_components=4, center=False).fit(self.X)
        self.assertTrue(pca.m_ is None)

        Z = np.random.normal(size=(20, self.X.shape[1]))
        V = pca.loadings_.values
        self.assertTrue(np.allclose(pca.predict_scores(Z), np.dot(Z, V)))

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
