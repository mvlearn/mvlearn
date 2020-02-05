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
# Implements multi-view kmeans clustering algorithm for data with 2-views.


import numpy as np
from .base_kmeans import BaseKMeans
from ..utils.utils import check_Xs
from sklearn.exceptions import NotFittedError, ConvergenceWarning
from scipy.spatial.distance import cdist


class MultiviewKMeans(BaseKMeans):

    '''
    An implementation of Multi-View K-Means using the co-EM algorithm.
    This algorithm currently handles two views of data.

    Parameters
    ---------
    n_clusters : int, optional, default=2
        The number of clusters

    random_state : int, optional, default=None
        Determines random number generation for initializing centroids.
        Can seed the random number generator with an int.

    patience : int, optional, default=5
        The number of EM iterations with no decrease in the objective
        function after which the algorithm will terminate.

    max_iter : int, optional, default=None
        The maximum number of EM iterations to run before
        termination.

    n_init : int, optional, default=5
        Number of times the k-means algorithm will run on different
        centroid seeds. The final result will be the best output of
        n_init runs with respect to total inertia across all views.

    init : {'k-means++', 'random'} or list of array-likes
        Method of initializing centroids, defaults to 'k-means++'.

        'k-means++': selects initial cluster centers for k-means clustering
        via a method that speeds up convergence.

        'random': choose n_cluster samples from the data for the initial
        centroids.

        If a list of array-likes is passed, the list should have a length of
        equal to the number of views. Each of the array-likes should have
        the shape

    Attributes
    ----------

    centroids_ : list of array-likes
        - centroids_ length: n_views
        - centroids_[i] shape: (n_clusters, n_features_i)
        The cluster centroids for each of the two views. centroids_[0]
        corresponds to the centroids of view 1 and centroids_[1] corresponds
        to the centroids of view 2.

    References
    ----------
    [1] Bickel S, Scheffer T (2004) Multi-view clustering. Proceedings of the
    4th IEEE International Conference on Data Mining, pp. 19–26
    '''

    def __init__(self, n_clusters=2, random_state=None, init='k-means++',
                 patience=5, max_iter=None, n_init=5, n_jobs=1):

        super().__init__()

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.patience = patience
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.centroids_ = None

    def _compute_dist(self, X, Y):
        return cdist(X, Y)
        
    def _init_centroids(self, Xs):

        '''
        Initializes the centroids for Multi-view KMeans or KMeans++ depending
        on which has been selected.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        centroids : list of array-likes
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)
            The cluster centroids for each of the two views. centroids[0]
            corresponds to the centroids of view 1 and centroids[1] corresponds
            to the centroids of view 2. These are not yet the final cluster
            centroids.
        '''
        centroids = None
        if self.init == 'random':
            # Random initialization of centroids
            indices = np.random.choice(Xs[0].shape[0], self.n_clusters)
            centers1 = Xs[0][indices]
            centers2 = Xs[1][indices]
            centroids = [centers1, centers2]
        elif self.init == 'k-means++':
            # Initializing centroids via kmeans++ implementation
            indices = list()
            centers2 = list()
            indices.append(np.random.randint(Xs[1].shape[0]))
            centers2.append(Xs[1][indices[0], :])

            # Compute the remaining n_cluster centroids
            for cent in range(self.n_clusters - 1):
                dists = self._compute_dist(centers2, Xs[1])
                dists = np.min(dists, axis=1)
                max_index = np.argmax(dists)
                indices.append(max_index)
                centers2.append(Xs[1][max_index])

            centers1 = Xs[0][indices]
            centers2 = np.array(centers2)
            centroids = [centers1, centers2]
        else:
            centroids = self.init
            try:
                centroids = check_Xs(centroids, enforce_views=2)
            except ValueError:
                msg = 'init must be a valid centroid initialization'
                raise ValueError(msg)
            for ind in range(len(centroids)):
                if centroids[ind].shape[0] != self.n_clusters:
                    msg = 'number of centroids per view must equal n_clusters'
                    raise ValueError(msg)
                if centroids[ind].shape[1] != Xs[0][0].shape[0]:
                    msg = ('feature dimensions of cluster centroids'
                           + 'must match those of data')
                    raise ValueError(msg)

        return centroids

    def _final_centroids(self, Xs, centroids):

        '''
        Computes the final cluster centroids based on consensus samples across
        both views. Consensus samples are those that are assigned to the same
        partition in both views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        centroids : list of array-likes
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)
            The cluster centroids for each of the two views. centroids[0]
            corresponds to the centroids of view 1 and centroids[1] corresponds
            to the centroids of view 2. These are not yet the final cluster
            centroids.
        '''

        # Compute consensus vectors for final clustering
        v1_consensus = list()
        v2_consensus = list()

        for clust in range(self.n_clusters):
            v1_distances = self._compute_dist(Xs[0], centroids[0])
            v1_partitions = np.argmin(v1_distances, axis=1).flatten()
            v2_distances = self._compute_dist(Xs[1], centroids[1])
            v2_partitions = np.argmin(v2_distances, axis=1).flatten()

            # Find data points in the same partition in both views
            part_indices = (v1_partitions == clust) * (v2_partitions == clust)

            # Recompute centroids based on these data points
            if (np.sum(part_indices) != 0):
                cent1 = np.mean(Xs[0][part_indices], axis=0)
                v1_consensus.append(cent1)

                cent2 = np.mean(Xs[1][part_indices], axis=0)
                v2_consensus.append(cent2)

        # Check if there are no consensus vectors
        self.centroids_ = [None, None]
        if (len(v1_consensus) == 0):
            msg = 'No distinct cluster centroids have been found.'
            raise ConvergenceWarning(msg)
        else:
            self.centroids_[0] = np.vstack(v1_consensus)
            self.centroids_[1] = np.vstack(v2_consensus)

            # Check if the number of consensus clusters is less than n_clusters
            if (self.centroids_[0].shape[0] < self.n_clusters):
                msg = ('Number of distinct cluster centroids ('
                       + str(self.centroids_[0].shape[0])
                       + ') found is smaller than n_clusters ('
                       + str(self.n_clusters)
                       + ').')
                raise ConvergenceWarning(msg)

            # Updates k if number of consensus clusters less than original
            # n_clusters value
            self.n_clusters = self.centroids_[0].shape[0]

    def _em_step(self, X, partition, centroids):

        n_samples = X.shape[0]
        new_centers = list()
        for cl in range(self.n_clusters):
            # Recompute centroids using samples from each cluster
            mask = (partition == cl)
            if (np.sum(mask) == 0):
                new_centers.append(centroids[cl])
            else:
                cent = np.mean(X[mask], axis=0)
                new_centers.append(cent)
        new_centers = np.vstack(new_centers)
            
        # Compute expectation and objective function
        distances = self._compute_dist(X, new_centers)
        new_parts = np.argmin(distances, axis=1).flatten()
        min_dists = distances[np.arange(n_samples), new_parts]
        o_funct = np.sum(min_dists)

        return new_parts, new_centers, o_funct

    def _preprocess_data(self, Xs):
        
        # Check that the input data is valid
        Xs = check_Xs(Xs, enforce_views=2)
        return Xs
    
    def fit(self, Xs):

        '''
        Fit the cluster centroids to the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        self : returns an instance of self.
        '''

        Xs = self._preprocess_data(Xs)
        
        # Type checking and exception handling for random_state parameter
        if self.random_state is not None:
            msg = 'random_state must be convertible to 32 bit unsigned integer'
            try:
                self.random_state = int(self.random_state)
            except TypeError:
                raise TypeError(msg)
            np.random.seed(self.random_state)

        # Type and value checking for n_clusters parameter
        if not (isinstance(self.n_clusters, int) and self.n_clusters > 0):
            msg = 'n_clusters must be a positive integer'
            raise ValueError(msg)

        # Type and value checking for patience parameter
        if not (isinstance(self.patience, int) and (self.patience > 0)):
            msg = 'patience must be a nonnegative integer'
            raise ValueError(msg)

        # Type and value checking for max_iter parameter
        max_iter = np.inf
        if self.max_iter is not None:
            if not (isinstance(self.max_iter, int) and (self.max_iter > 0)):
                msg = 'max_iter must be a positive integer'
                raise ValueError(msg)
            max_iter = self.max_iter

        # Type and value checking for n_init parameter
        if not (isinstance(self.n_init, int) and (self.n_init > 0)):
            msg = 'n_init must be a nonnegative integer'
            raise ValueError(msg)

        # If initial centroids passed in, then n_init should be 1
        n_init = self.n_init
        if not isinstance(self.n_init, str):
            n_init = 1

        # Run multi-view kmeans for n_init different centroid initializations
        min_inertia = np.inf
        best_centroids = None

        for _ in range(n_init):

            # Initialize centroids for clustering
            centroids = self._init_centroids(Xs)

            # Initializing partitions, objective value, and loop vars
            distances = self._compute_dist(Xs[1], centroids[1])
            parts = np.argmin(distances, axis=1).flatten()
            partitions = [None, parts]
            objective = [np.inf, np.inf]
            iter_stall = 0
            iter_num = 0

            # While objective is still decreasing and iterations < max_iter
            while(iter_stall < self.patience and iter_num < max_iter):
                iter_num += 1
                pre_view = (iter_num) % 2
                view = (iter_num + 1) % 2
                # Switch partitions and compute maximization
                
                partitions[view], centroids[view], o_funct = self._em_step(
                    Xs[view], partitions[pre_view], centroids[view]) 
                
                # Track the number of iterations without improvement
                if(o_funct < objective[view]):
                    objective[view] = o_funct
                    iter_stall = 0
                else:
                    iter_stall += 1

            # Update min_intertia and best centroids if lower intertia
            total_inertia = np.sum(objective)
            if(total_inertia < min_inertia or best_centroids is None):
                min_inertia = total_inertia
                best_centroids = centroids

        # Compute final cluster centroids
        self._final_centroids(Xs, best_centroids)

        return self

    def predict(self, Xs):

        '''
        Predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two
            views of the data. The two views can each have a different
            number of features, but they must have the same number of samples.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''

        Xs = check_Xs(Xs, enforce_views=2)

        # Check whether or not centroids were properly fitted
        if self.centroids_ is None:
            msg = 'This MultiviewKMeans instance is not fitted yet.'
            raise NotFittedError(msg)

        if self.centroids_[0] is None:
            msg = 'This MultiviewKMeans instance has no cluster centroids.'
            raise AttributeError(msg)

        dist1 = self._compute_dist(Xs[0], self.centroids_[0])
        dist2 = self._compute_dist(Xs[1], self.centroids_[1])
        dist_metric = dist1 + dist2
        predictions = np.argmin(dist_metric, axis=1).flatten()

        return predictions
