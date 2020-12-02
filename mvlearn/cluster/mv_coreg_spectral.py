# License: MIT
#
# Implements multi-view spectral clustering algorithm for data with
# multiple views.


import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

from .mv_spectral import MultiviewSpectralClustering

AFFINITY_METRICS = ['rbf', 'nearest_neighbors', 'poly']


class MultiviewCoRegSpectralClustering(MultiviewSpectralClustering):

    r'''
    An implementation of co-regularized multi-view spectral clustering based on
    an unsupervied version of the co-training framework.
    This algorithm uses the pairwise co-regularization scheme as described
    in [#4Clu]_. This algorithm can handle 2 or more views of data.

    Parameters
    ----------
    n_clusters : int
        The number of clusters

    v_lambda : float, optional, default=2
        The regularization parameter. This parameter trades-off the spectral
        clustering objectives with the degree of agreement between each pair
        of views in the new representation. Must be a positive value.

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

    Attributes
    ----------
    labels_ : array-like, shape (n_samples,)
        Cluster labels for each point.

    embedding_ : array-like, shape (n_samples, n_clusters)
        The final spectral representation of the data to be used as input
        for the KMeans clustering step.

    objective_ : array-like, shape (n_views, n_iterations)
        The value of the spectral clustering objective for each view at
        the end of each iteration.

    Notes
    -----

    In standard spectral clustering, the eigenvector matrix U for a given view
    is the new data representation to be used for the subsequent k-means
    clustering stage. In this algorithm, the objective function has been
    altered to encourage the pairwise similarities of examples under the new
    representation to be similar across all views.

    The modified spectral clustering objective for the case of two views is
    shown and derived in [#4Clu]. In the clustering objective, the
    hyperparameter lambda trades-off the spectral clustering objectives and
    the disagreement term.

    For a fixed lambda and n, the objective function is bounded from above and
    non-decreasing. As such, the algorithm is guaranteed to converge.

    References
    ----------
    .. [#4Clu] Abhishek Kumar, Piyush Rai, and Hal Daume.  Co-regularized
            multi-view spectral cluster-ing. In Proceedings of the 24th
            International Conference on Neural Information Processing Systems,
            page 1413–1421. Curran Associates Inc., 2011.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.cluster import MultiviewCoRegSpectralClustering
    >>> from sklearn.metrics import normalized_mutual_info_score as nmi_score
    >>> # Get 5-class data
    >>> data, labels = load_UCImultifeature(select_labeled = list(range(5)))
    >>> mv_data = data[:2]  # first 2 views only
    >>> mv_spectral = MultiviewCoRegSpectralClustering(n_clusters=5,
    ...     random_state=10, n_init=100)
    >>> mv_clusters = mv_spectral.fit_predict(mv_data)
    >>> nmi = nmi_score(labels, mv_clusters, average_method='arithmetic')
    >>> print('{0:.3f}'.format(nmi))
    0.663
    '''

    def __init__(self, n_clusters=2, v_lambda=2, random_state=None,
                 info_view=None, max_iter=10, n_init=10, affinity='rbf',
                 gamma=None, n_neighbors=10):

        super().__init__(n_clusters=n_clusters, random_state=random_state,
                         info_view=info_view, max_iter=max_iter,
                         n_init=n_init, affinity=affinity, gamma=gamma,
                         n_neighbors=n_neighbors)

        self.v_lambda = v_lambda
        self.objective_ = None

    def _init_umat(self, X):

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
         u_mat : array-like, shape(n_samples, n_clusters)
            The top n_cluster eigenvectors of the normalized graph
            laplacian.

        laplacian : array-like, shape(n_samples, n_samples)
            The normalized graph laplacian for the similarity matrix.

        obj_val : float
            The updated value for the objective function for the given
            view.

        '''

        # Compute the normalized laplacian
        d_mat = np.diag(np.sum(X, axis=1))
        d_alt = np.sqrt(np.linalg.inv(d_mat))
        laplacian = d_alt @ X @ d_alt

        # Make the resulting matrix symmetric
        laplacian = (laplacian + np.transpose(laplacian)) / 2.0

        # Obtain the top n_cluster eigenvectors of the laplacian
        u_mat, d_mat, _ = sp.sparse.linalg.svds(laplacian, k=self.n_clusters)
        obj_val = np.sum(d_mat)

        return u_mat, laplacian, obj_val

    def fit(self, Xs):

        r'''
        Performs clustering on the multiple views of data.

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
        self : returns an instance of self.
        '''

        check_u_mats = list()

        # Perform checks on inputted parameters and data
        Xs = self._param_checks(Xs)
        if self.v_lambda <= 0:
            msg = 'v_lambda must be a positive value'
            raise ValueError(msg)

        # Compute the similarity matrices
        sims = [self._affinity_mat(X) for X in Xs]

        # Initialize matrices of eigenvectors
        U_mats = []
        L_mats = []
        obj_vals = np.zeros((self._n_views, self.max_iter))
        for ind in range(len(sims)):
            u_mat, l_mat, o_val = self._init_umat(sims[ind])
            U_mats.append(u_mat)
            L_mats.append(l_mat)
            obj_vals[ind, 0] = o_val

        check_u_mats.append(U_mats[0])

        # Iteratively solve for all U's
        n_items = Xs[0].shape[0]
        for it in range(1, self.max_iter):

            # Performing alternating maximization by cycling through all
            # pairs of views and updating all except view 1
            for v1 in range(1, self._n_views):

                # Computing the regularization term for view v1
                l_comp = np.zeros((n_items, n_items))
                for v2 in range(self._n_views):
                    if v1 != v2:
                        l_comp = l_comp + U_mats[v2] @ U_mats[v2].T
                l_comp = (l_comp + l_comp.T) / 2

                # Adding the symmetrized graph laplacian for view v1
                l_mat = L_mats[v1] + self.v_lambda * l_comp
                U_mats[v1], d_mat, _ = sp.sparse.linalg.svds(l_mat,
                                                             k=self.n_clusters)
                obj_vals[v1, it] = np.sum(d_mat)

            # Update U and the objective function value for view 1
            l_comp = np.zeros((n_items, n_items))
            for vi in range(self._n_views):
                if vi != 0:
                    l_comp = l_comp + U_mats[vi] @ U_mats[vi].T
            l_comp = (l_comp + l_comp.T) / 2
            l_mat = L_mats[0] + self.v_lambda * l_comp
            U_mats[0], d_mat, _ = sp.sparse.linalg.svds(l_mat,
                                                        k=self.n_clusters)
            obj_vals[0, it] = np.sum(d_mat)
            check_u_mats.append(U_mats[0])
        self.objective_ = obj_vals

        # Create final spectral embedding to cluster
        V_mat = np.hstack(U_mats)
        norm_v = np.sqrt(np.diag(V_mat @ V_mat.T))
        norm_v[norm_v == 0] = 1
        self.embedding_ = np.linalg.inv(np.diag(norm_v)) @ V_mat

        # Perform kmeans clustering with embedding
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init,
                        random_state=self.random_state)
        self.labels_ = kmeans.fit_predict(self.embedding_)
        return self

    def fit_predict(self, Xs, y=None):

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

        y : ignored
            Included for API compliance.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.
        '''

        self.fit(Xs)
        labels = self.labels_
        return labels
