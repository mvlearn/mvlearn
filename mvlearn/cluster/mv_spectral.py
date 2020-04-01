
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
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.neighbors import NearestNeighbors

AFFINITY_METRICS = ['rbf', 'nearest_neighbors', 'poly']


class MultiviewSpectralClustering(BaseEstimator):

    r'''
    An implementation of multi-view spectral clustering using the
    basic co-training framework as described in [#1Clu]_.
    Additionally, this can be effective when the dataset naturally
    contains features that are of 2 different data types, such as
    continuous features and categorical features [#3Clu]_, and then the
    original features are separated into two views in this way.

    This algorithm can handle 2 or more views of data.

    Parameters
    ----------
    n_clusters : int
        The number of clusters

    n_views : int, optional, default=2
        The number of different views of data.

    random_state : int, optional, default=None
        Determines random number generation for k-means.

    info_view : int, optional, default=None
        The most informative view. Must be between 0 and n_views-1
        If given, then the final clustering will be performed on the
        designated view alone. Otherwise, the algorithm will concatenate
        across all views and cluster on the result.

    max_iter : int, optional, default=10
        The maximum number of iterations to run the clustering
        algorithm.

    n_init : int, optional, default=10
        The number of random initializations to use for k-means clustering.

    affinity : string, optional, default='rbf'
        The affinity metric used to construct the affinity matrix. Options
        include 'rbf' (radial basis function), 'nearest_neighbors', and
        'poly' (polynomial)

    gamma : float, optional, default=None
        Kernel coefficient for rbf and polynomial kernels. If None then
        gamma is computed as 1 / (2 * median(pair_wise_distances(X))^2)
        for each data view X.

    n_neighbors : int, optional, default=10
        Only used if nearest neighbors is selected for affinity. The
        number of neighbors to use for the nearest neighbors kernel.

    Notes
    -----
    Multi-view spectral clustering adapts the spectral clustering algorithm
    to applications where more than one view of data is available. This
    algorithm relies on the basic assumptions of the co-training, which are:
    (a) Sufficiency: each view is sufficient for classification on its own,
    (b) Compatibility: the target functions in both views predict the same
    labels for co-occurring features with high probability, and (c)
    Conditional independence: the views are conditionally independent given
    the class labels. In contrast to multi-view k-means clustering,
    multi-view spectral clustering performs well on arbitrary shaped clusters,
    and can therefore be readily used in applications where clusters are not
    expected to be convex. However multi-view spectral clustering tends to be
    computationally expensive unless the similarity graph for the data is
    sparse.

    Multi-view spectral clustering works by using the spectral embedding
    from one view to constrain the similarity graph in the other view. By
    iteratively applying this procedure, the clustering of the two views
    tend to each other. Here we outline the algorithm for the Multi-view
    Spectral clustering algorithm for 2 views.

    |

    *Multi-view Spectral Clustering Algorithm (for 2 views)*

    Input: Similarity matrix for both views: :math:`\mathbf{K}_1, \mathbf{K}_2`

    Output: Assignments to k clusters

        #. Initialize: :math:`\mathbf{L}_v = \mathbf{D}_v^{-1/2}
           \mathbf{K}_v\mathbf{D}_v^{-1/2}` for :math:`v = 1, 2`
            :math:`\mathbf{U}_v^0` is an :math:`n \times k` matrix with the
            top k eigenvectors of :math:`\mathbf{L}_v` for :math:`v = 1, 2`

        #. For :math:`i = 1` to iter:

            a. :math:`\mathbf{S}_1 = sym(\mathbf{U}_2^{i-1}
               {\mathbf{U}_2^{i-1}}^T\mathbf{K}_1)`

            b. :math:`\mathbf{S}_2 = sym(\mathbf{U}_1^{i-1}
               {\mathbf{U}_1^{i-1}}^T\mathbf{K}_2)`

            c. Use :math:`\mathbf{S}_1` and :math:`\mathbf{S}_2` as the new
               graph similarities and compute the Laplacians. Solve for the
               largest k eigenvectors to obtain :math:`\mathbf{U}_1^i` and
               :math:`\mathbf{U}_2^i`.

        #. Row-normalize :math:`\mathbf{U}_1^i` and :math:`\mathbf{U}_2^i`.

        #. Form matrix :math:`\mathbf{V} = \mathbf{U}_v^i`, where :math:`v` is
           believed to be the most informative view a priori. If there is no
           prior knowledge on the view informativeness, matrix
           :math:`\mathbf{V}` can also be set to the column-wise concatenation
           of the two :math:`\mathbf{U}_v^i` s.

        #. Assign example j to cluster c if the j-th row of :math:`\mathbf{V}`
           is assigned to cluster c by the k-means algorithm.


    References
    ----------
    .. [#1Clu] Abhishek Kumar and Hal Daume. A Co-training Approach for
            Multiview Spectral Clustering. In International Conference
            on Machine Learning, 2011

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.cluster import MultiviewSpectralClustering
    >>> from sklearn.metrics import normalized_mutual_info_score as nmi_score
    >>> # Get 5-class data
    >>> data, labels = load_UCImultifeature(select_labeled = list(range(5)))
    >>> mv_data = data[:2]  # first 2 views only
    >>> mv_spectral = MultiviewSpectralClustering(n_clusters=5,
            random_state=10, n_init=100)
    >>> mv_clusters = mv_spectral.fit_predict(mv_data)
    >>> nmi = nmi_score(labels, mv_clusters)
    >>> print('{0:.3f}\n'.format(nmi))
    '0.872'
    '''
    def __init__(self, n_clusters=2, n_views=2, random_state=None,
                 info_view=None, max_iter=10, n_init=10, affinity='rbf',
                 gamma=None, n_neighbors=10):

        super().__init__()

        if not (isinstance(n_clusters, int) and n_clusters > 0):
            msg = 'n_clusters must be a positive integer'
            raise ValueError(msg)

        if not (isinstance(n_views, int) and n_views > 1):
            msg = 'n_views must be a positive integer greater than 1'
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

        if not (isinstance(max_iter, int) and (max_iter > 0)):
            msg = 'max_iter must be a positive integer'
            raise ValueError(msg)

        if not (isinstance(n_init, int) and n_init > 0):
            msg = 'n_init must be a positive integer'
            raise ValueError(msg)

        if affinity not in AFFINITY_METRICS:
            msg = 'affinity must be a valid affinity metric'
            raise ValueError(msg)

        if gamma is not None and not ((isinstance(gamma, float) or
                                       isinstance(gamma, int)) and gamma > 0):
            msg = 'gamma must be a positive float'
            raise ValueError(msg)

        if not (isinstance(n_neighbors, int) and n_neighbors > 0):
            msg = 'n_neighbors must be a positive integer'
            raise ValueError(msg)

        self.n_clusters = n_clusters
        self.n_views = n_views
        self.random_state = random_state
        self.info_view = info_view
        self.max_iter = max_iter
        self.n_init = n_init
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def _affinity_mat(self, X):

        r'''
        Computes the affinity matrix based on the selected
        kernel type.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix from which we will compute the
            affinity matrix.

        Returns
        -------
        sims : array-like, shape (n_samples, n_samples)
            The resulting affinity kernel.

        '''

        sims = None

        # If gamma is None, then compute default gamma value for this view
        gamma = self.gamma
        if self.gamma is None:
            distances = cdist(X, X)
            gamma = 1 / (2 * np.median(distances) ** 2)
        # Produce the affinity matrix based on the selected kernel type
        if (self.affinity == 'rbf'):
            sims = rbf_kernel(X, gamma=gamma)
        elif(self.affinity == 'nearest_neighbors'):
            neighbor = NearestNeighbors(n_neighbors=self.n_neighbors)
            neighbor.fit(X)
            sims = neighbor.kneighbors_graph(X).toarray()
        else:
            sims = polynomial_kernel(X, gamma=gamma)

        return sims

    def _compute_eigs(self, X):

        r'''
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
        u_mat, _, _ = sp.sparse.linalg.svds(laplacian, k=self.n_clusters)
        la_eigs = u_mat[:, :self.n_clusters]
        return la_eigs

    def fit_predict(self, Xs):

        r'''
        Performs clustering on the multiple views of data and returns
        the cluster labels.

        Parameters
        ----------

        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size n_views, corresponding to the number
            of views of data. Each view can have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.
        '''

        Xs = check_Xs(Xs)
        if len(Xs) != self.n_views:
            msg = 'Length of Xs must be the same as n_views'
            raise ValueError(msg)

        # Compute the similarity matrices
        sims = [self._affinity_mat(X) for X in Xs]

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

        # Performing k-means clustering
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
