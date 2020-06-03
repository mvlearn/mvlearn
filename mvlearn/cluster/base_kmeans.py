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
# A base class for multi-view kmeans clustering algorithms that apply the
# co-EM framework.

import numpy as np
from abc import abstractmethod
from .base import BaseCluster


class BaseKMeans(BaseCluster):
    '''
    A base class for kmeans clustering.
    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    '''

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs):

        '''
        A method to fit clustering parameters to the multiview data.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different views to fit the model on.

        Returns
        -------
        self :  returns and instance of self.
        '''

        return self

    @abstractmethod
    def predict(self, Xs):

        '''
        A method to predict cluster labels of multiview data.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different views to cluster.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            Returns the predicted cluster labels for each sample.
        '''
        return

    def fit_predict(self, Xs):

        '''
        Fit the cluster centroids to the data and then
        predict the cluster labels for the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size 2, corresponding to the two views
            of the data. The two views can each have a different number
            of features, but they must have the same number of samples.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.

        '''

        self.fit(Xs)
        labels = self.predict(Xs)
        return labels
