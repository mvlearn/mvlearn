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
from .base_cluster import BaseCluster
from ..utils.utils import check_Xs
from sklearn.exceptions import NotFittedError


class MultiviewKMeans(BaseCluster):

    '''
    An implementation of Multi-View K-Means using the co-EM algorithm.
    This algorithm currently handles two views of data.

    Paramters
    ---------
    k : int
        The number of clusters

    random_state : int (default=None)
        Determines random number generation for initializing centroids.
        Can seed the random number generator with an int.

    Attributes
    ----------

    _k : int
        The number of clusters

    _random_state : int
        The seed for the random number generator used during centroid
        initialization.

    _centroids : list of array_likes
        - _centroids shape: (2,)
        - _centroids[0] shape: (n_clusters, n_features_i)
        The cluster centroids for each of the two views. _centroids[0]
        corresponds to the centroids of view 1 and _centroids[1] corresponds
        to the centroids of view 2.

    References
    ----------
    [1] Bickel S, Scheffer T (2004) Multi-view clustering. Proceedings of the
    4th IEEE International Conference on Data Mining, pp. 19–26
    '''

    def __init__(self, k=5, random_state=None):

        super().__init__()

        if not (isinstance(k, int) and k > 0):
            msg = 'k must be a positive integer'
            raise ValueError(msg)

        if random_state is not None:
            msg = 'random_state must be convertible to 32 bit unsigned integer'
            try:
                random_state = int(random_state)
            except ValueError:
                raise ValueError(msg)
            np.random.seed(random_state)

        self._centroids = None
        self._k = k
        self._random_state = random_state

    def _compute_distance(self, X, centers):

        '''
        Computes the Euclidean distance between each sample point
        in the given view and each cluster centroid.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data from a single view.
        centers : array_like, shape (n_clusters, n_features)
            The cluster centroids for a single view.

        Returns
        -------
        distances : array-like (n_clusters, n_samples)
            An array of Euclidean distances between each
            sample point and each cluster centroid.

        '''

        distances = list()

        for cl in range(self._k):
            dist = X - centers[cl]
            dist = np.linalg.norm(dist, axis=1)
            distances.append(dist)

        distances = np.vstack(distances)
        return distances

    def fit(self, Xs, patience=5, max_iter=None):

        '''
        Fit the cluster centroids to the data.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (2,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        patience: int, optional (default=5)
            The number of EM iterations with no decrease in the objective
            function after which the algorithm will terminate.

        max_iter: int, optional (default=None)
            The maximum number of EM iterations to run before
            termination.

        '''

        Xs = check_Xs(Xs, enforce_views=2)

        if not (isinstance(patience, int) and (patience > 0)):
            msg = 'patience must be a nonnegative integer'
            raise ValueError(msg)

        if max_iter is not None:
            if not (isinstance(max_iter, int) and (max_iter > 0)):
                msg = 'max_iter must be a positive integer'
                raise ValueError(msg)
        else:
            max_iter = np.inf

        # Random initialization of centroids
        indices1 = np.random.choice(Xs[0].shape[0], self._k)
        centers1 = Xs[0][indices1]
        indices2 = np.random.choice(Xs[1].shape[0], self._k)
        centers2 = Xs[1][indices2]
        self._centroids = [centers1, centers2]

        # Initializing partitions, objective function value, and loop variables
        distances = self._compute_distance(Xs[1], centers2)
        parts = np.argmin(distances, axis=0).flatten()
        partitions = [None, parts]
        objective = [np.inf, np.inf]
        iter_stall = 0
        iter_num = 0
        entropy = 0

        # While objective is still decreasing and num of iterations < max_iter
        while(iter_stall < patience and iter_num < max_iter):
            iter_num += 1
            pre_view = (iter_num) % 2
            view = (iter_num + 1) % 2

            # Switch partitions and compute maximization
            new_centers = list()
            for cl in range(self._k):
                # Isolate data points from each cluster to recompute centroids
                mask = (partitions[pre_view] == cl)
                if (np.sum(mask) == 0):
                    new_centers.append(self._centroids[view][cl])
                else:
                    cent = np.mean(Xs[view][mask], axis=0)
                    new_centers.append(cent)
            self._centroids[view] = np.vstack(new_centers)

            # Compute expectation
            distances = self._compute_distance(Xs[view], self._centroids[view])
            partitions[view] = np.argmin(distances, axis=0).flatten()

            # Recompute the objective function
            o_funct = 0
            for cl in range(self._k):
                # Collect data points in each cluster and compute within
                # cluster distances
                vecs = Xs[view][(partitions[view] == cl)]
                dist = np.linalg.norm(vecs - self._centroids[view][cl], axis=1)
                o_funct += np.sum(dist)

            # Track of number of iterations without improvement
            if(o_funct < objective[view]):
                objective[view] = o_funct
                iter_stall = 0
            else:
                iter_stall += 1

        # Compute consensus vectors for final clustering
        v1_consensus = list()
        v2_consensus = list()
        for clust in range(self._k):
            v1_distances = self._compute_distance(Xs[0], self._centroids[0])
            v1_partitions = np.argmin(v1_distances, axis=0).flatten()
            v2_distances = self._compute_distance(Xs[1], self._centroids[1])
            v2_partitions = np.argmin(v2_distances, axis=0).flatten()

            # Find data points in the same partition in both views
            part_indices = (v1_partitions == clust) * (v2_partitions == clust)

            # Recompute centroids based on these data points
            if (np.sum(part_indices) != 0):
                cent1 = np.mean(Xs[0][part_indices], axis=0)
                v1_consensus.append(cent1)

                cent2 = np.mean(Xs[1][part_indices], axis=0)
                v2_consensus.append(cent2)
        
        self._centroids[0] = np.vstack(v1_consensus)
        self._centroids[1] = np.vstack(v2_consensus)
        # Updates k if number of consensus clusters less than original k value
        self._k = self._centroids[0].shape[0]

        return self

    def predict(self, Xs):

        '''
        Predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (2,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two
            views of the data. The two views can each have a different
            number of features, but they must have the same number of samples.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''

        Xs = check_Xs(Xs, enforce_views=2)

        if self._centroids is None:
            msg = 'This MultiviewKMeans instance is not fitted yet.'
            raise NotFittedError(msg)

        dist1 = self._compute_distance(Xs[0], self._centroids[0])
        dist2 = self._compute_distance(Xs[1], self._centroids[1])
        dist_metric = dist1 + dist2
        predictions = np.argmin(dist_metric, axis=0).flatten()

        return predictions

    def fit_predict(self, Xs, patience=5, max_iter=None):

        '''
        Fit the cluster centroids to the data and then
        predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (2,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views
            of the data. The two views can each have a different number
            of features, but they must have the same number of samples.

        patience: int, optional (default=5)
            The number of EM iterations with no decrease in the objective
            function after which the algorithm will terminate.

        max_iter: int, optional (default=None)
            The maximum number of EM iterations to run before
            termination.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''

        self.fit(Xs, patience, max_iter)
        predictions = self.predict(Xs)
        return predictions
