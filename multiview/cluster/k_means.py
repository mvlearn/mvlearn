import numpy as np
from .base_cluster import BaseCluster
from ..utils.cluster_utils import check_Xs

class KMeans(BaseCluster):

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

    max_iter : int (default=None)
        The maximum number of iterations to run the EM algorithm.

    Attributes
    ----------

    _k : int
        The number of clusters

    _random_state : int
        The seed for the random number generator used during centroid
        initialization.

    _max_iter : int
        The maximum number of iterations to run the EM algorithm.

    _centroids : list of array_likes
        - _centroids shape: (2,)
        - _centroids[0] shape: (n_clusters, n_features_i)
        The cluster centroids for each of the two views. _centroids[0]
        corresponds to the centroids of view 1 and _cnetroids[1] corresponds
        to the centroids of view 2.


    '''

    def __init__(self, k=5, random_state=None, max_iter=None):

        super().__init__() 
        
        if not isinstance(k, int):
            msg = 'k must be a positive integer'
            raise ValueError(msg)
        if (k < 1):    
            msg = 'k must be a positive integer'
            raise ValueError(msg)
        
        if max_iter is not None:
            msg = 'max_iter must be a positive integer'
            if not isinstance(max_iter, int):
                raise ValueError(msg)
            if (max_iter < 1):
                raise ValueError(msg)
            self._max_iter = max_iter

        if random_state is not None:
            msg = 'random_state must be convertible to a 32 bit unsigned integer'
            try:
                random_state = int(random_state)
            except ValueError:
                raise ValueError(msg)
            np.random.seed(random_state)
        
        self._centroids = None
        self._k = k
        self._random_state = random_state
        self._max_iter = np.inf
        
    def _compute_distance(self, X, centers):

        '''
        Computes the euclidean distance between each sample point
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
            An array of euclidean distances between each
            sample point and each cluster centroid.

        '''

        distances = list()
        for cl in range(self._k):
            dist = X - centers[cl]
            dist = np.linalg.norm(dist, axis=1)
            distances.append(dist)

        distances = np.vstack(distances)
        return distances

    def fit(self, Xs, patience=5):

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

        '''
        
        Xs = check_Xs(Xs)

        if not isinstance(patience, int):
            msg = 'patience must be a nonegative integer'
            raise ValueError(msg)
        if (patience < 0):
            msg = 'patience must be a nonnegative integer'
            raise ValueError(msg)
        
        indices = np.random.choice(Xs[1].shape[0], self._k)
        centers = Xs[1][indices]
        self._centroids = [None, centers]

        distances = self._compute_distance(Xs[1], centers)
        parts = np.argmin(distances, axis=0).flatten()
        partitions = [None, parts]
        objective = [np.inf, np.inf]
        iter_stall = 0
        iter_num = 0
        entropy = 0

        while(iter_stall < patience and iter_num < self._max_iter):
            iter_num += 1
            view = (iter_num + 1) % 2

            # Switch partitions, maximization, and expectation

            new_centers = list()
            for cl in range(self._k):
                mask = (partitions[(view + 1) % 2] == cl)
                if (np.sum(mask) == 0):
                    new_centers.append(self._centroids[view][cl])
                else:
                    cent = np.mean(Xs[view][mask], axis=0)
                    new_centers.append(cent)
            self._centroids[view] = np.vstack(new_centers)

            distances = self._compute_distance(Xs[view], self._centroids[view])
            new_partitions = np.argmin(distances, axis=0).flatten()

            # Recompute the objective function
            o_funct = 0
            for clust in range(self._k):
                vecs = Xs[view][(partitions[view] == clust)]
                dist = np.linalg.norm(vecs - self._centroids[view][clust], axis=1)
                o_funct += np.sum(dist)

            # Track of number of iterations without improvement
            if(o_funct < objective[view]):
                objective[view] = o_funct
                iter_stall = 0
            else:
                iter_stall += 1
                
        v1_consensus = list()
        v2_consensus = list()

        for clust in range(self._k):

            v1_distances = self._compute_distance(Xs[0], self._centroids[0])
            v1_partitions = np.argmin(v1_distances, axis=0).flatten()
            v2_distances = self._compute_distance(Xs[1], self._centroids[1])
            v2_partitions = np.argmin(v2_distances, axis=0).flatten()

            part_indices = (v1_partitions == clust) * (v2_partitions == clust)

            if (np.sum(part_indices) != 0):
                cent1 = np.mean(Xs[0][part_indices], axis=0)
                v1_consensus.append(cent1)
                
                cent2 = np.mean(Xs[1][part_indices], axis=0)
                v2_consensus.append(cent2)

        self._centroids[0] = np.vstack(v1_consensus)
        self._centroids[1] = np.vstack(v2_consensus)
                
        return self

    def predict(self, Xs):

        '''
        Predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (2,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of the data.
            The two views can each have a different number of features, but they must
            have the same number of samples.


        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''
        
        Xs = check_Xs(Xs)

        dist1 = self._compute_distance(Xs[0], self._centroids[0])
        dist2 = self._compute_distance(Xs[1], self._centroids[1])
        dist_metric = dist1 + dist2
        predictions = np.argmin(dist_metric, axis=0).flatten()

        return predictions

    def fit_predict(self, Xs, patience=5):

        '''
        Fit the cluster centroids to the data and then
        predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array_likes
            - Xs shape: (2,)
            - Xs[0] shape: (n_samples, n_features_i)
            This list must be of size 2, corresponding to the two views of the data.
            The two views can each have a different number of features, but they must
            have the same number of samples.


        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''

        self.fit(Xs)
        partitions = self.predict(Xs)
        return partitions
