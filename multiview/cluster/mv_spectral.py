# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implements multi-view spectral clustering algorithm for data with
# multiple views.


import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from ..utils.utils import check_Xs
from sklearn.exceptions import NotFittedError


class MultiviewSpectralClustering(BaseEstimator):

    '''
    An implementation of Multi-View Spectral using the
    basic co-training framework.

    Paramters
    ---------
    n_clusters : int
        The number of clusters

    n_views : int, optional (default=2)
        The number of different views of data.

    random_state : int, optional (default=None)
        Determines random number generation for kmeans.

    info_view : int, optional (default=None)
        The most informative view. Must be between 0 and n_views-1
        If given, then the final clustering will be performed on the
        designated view alone. Otherwise, the algorithm will concatenate
        across all views and cluster on the result.

    max_iter : int, optional (default=10)
        The maximum number of iterations to run the clustering
        algorithm.

    References
    ----------
    [1] Abhishek Kumar and Hal Daume. A Co-training Approach for Multiview
    Spectral Clustering. In International Conference on Machine Learning, 2011

    '''

    def __init__(self, n_clusters, n_views=2, random_state=None,
                 info_view=None, n_iter=10):

        super().__init__()

        if not (isinstance(n_clusters, int) and n_clusters > 0):
            msg = 'n_clusters must be a positive integer'
            raise ValueError(msg)

        if not (isinstance(n_views, int) and n_views > 0):
            msg = 'n_views must be a positive integer'
            raise ValueError(msg)

        if random_state is not None:
            msg = 'random_state must be convertible to 32 bit unsigned integer'
            try:
                random_state = int(random_state)
            except ValueError:
                raise ValueError(msg)
            np.random.seed(random_state)

        self.info_view = None
        if info_view is not None:
            if (isinstance(info_view, int)
                    and (info_view >= 0 and info_view < n_views)):
                self.info_view = info_view
            else:
                msg = 'info_view must be an integer between 0 and n_clusters-1'
                raise ValueError(msg)

        if not (isinstance(n_iter, int) and (n_iter > 0)):
            msg = 'max_iter must be a positive integer'
            raise ValueError(msg)

        self.n_clusters = n_clusters
        self.n_views = n_views
        self.random_state = random_state
        self.info_view = info_view
        self.max_iter = n_iter

    def _gaussian_sim(self, X):

        '''
        Computes the gaussian similarity kernel for a given matrix.
        The sigma used is the median pairwise distances.

        Parameters
        ----------
        X : array_like, shape(n_samples, n_features)
            The data matrix from which we will compute the gaussian
            similarity kernel.

        Returns
        -------
        sims : array_like, shape(n_samples, n_samples)
            The gaussian similarity kernel.

        '''

        distances = cdist(X, X)
        sq_dists = np.square(distances)
        norm_dists = sq_dists / (2 * np.median(sq_dists))
        sims = np.exp(-norm_dists)

        return sims

    def _compute_eigs(self, X):

        '''
        Computes the top several eigenvectors of the
        normalized graph laplacian of a given similarity matrix.
        The number of eigenvectors returned is equal to n_clusters.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_samples)
            The similarity matrix for the data in a single view.

        Returns
        -------
        la_eigs : array-like, shape(n_samples, n_clusters)
            The top n_cluster eigenvectors of the normalized graph
            laplacian.
        '''

        # Compute the normalized laplacian
        d_mat = np.diag(np.sum(X, axis=1))
        d_alt = np.sqrt(np.linalg.inv(np.abs(d_mat)))
        laplacian = d_alt @ X @ d_alt

        # Make the resulting matrix symmetric
        laplacian = (laplacian + np.transpose(laplacian)) / 2.0

        # Obtain the top n_cluster eigenvectors of the laplacian
        e_vals, e_vecs = np.linalg.eig(laplacian)
        indices = np.argsort(np.real(e_vals))[-self.n_clusters:]
        la_eigs = np.real(e_vecs[:, indices])

        return la_eigs

    def fit_predict(self, Xs):

        '''
        Performs clustering on the multiple views of data and returns
        the cluster labels.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (n_views,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size n_views, corresponding to the number
            of views of data. Each view can have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        predictions : array_like, shape(n_samples,)
            The predicted cluster labels for each sample.
        '''

        Xs = check_Xs(Xs)
        if len(Xs) != self.n_views:
            msg = 'Length of Xs must be the same as n_views'
            raise ValueError(msg)

        # Compute the similarity matrices
        sims = [self._gaussian_sim(X) for X in Xs]

        # Initialize matrices of eigenvectors
        U_mats = [self._compute_eigs(sim) for sim in sims]

        # Iteratively compute new graph similarities, laplacians,
        # and eigenvectors
        for iter in range(self.max_iter):

            # Compute the sums of the products of the spectral embeddings
            # and their transposes
            eig_sums = [u_mat @ np.transpose(u_mat) for u_mat in U_mats]
            U_sum = np.sum(np.array(eig_sums), axis=0)
            new_sims = list()

            for view in range(self.n_views):
                # Compute new graph similarity representation
                mat1 = sims[view] @ (U_sum - eig_sums[view])
                mat1 = (mat1 + np.transpose(mat1)) / 2.0
                new_sims.append(mat1)
                # Recompute eigenvectors
                U_mats = [self._compute_eigs(sim)
                          for sim in new_sims]

        # Row normalize
        for view in range(self.n_views):
            U_norm = np.linalg.norm(U_mats[view], axis=1).reshape((-1, 1))
            U_norm[U_norm == 0] = 1
            U_mats[view] /= U_norm

        # Performing kmeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state)
        predictions = None
        if self.info_view is not None:
            # Use a single view if one was previously designated
            predictions = kmeans.fit_predict(U_mats[self.info_view])
        else:
            # Otherwise, perform columwise concatenation across views
            # and use result for clustering
            V_mat = np.hstack(U_mats)
            predictions = kmeans.fit_predict(V_mat)

        return predictions
