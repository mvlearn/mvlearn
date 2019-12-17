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
# A base class for multi-view clustering algorithms that apply the
# co-EM framework.

import numpy as np
from abc import abstractmethod
from sklearn.base import BaseEstimator


class BaseCluster(BaseEstimator):
    '''
    A base class for clustering multiview data.
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

    @abstractmethod
    def fit_predict(self, Xs):

        '''
        A method to fit clustering parameters and predict cluster
        labels of multiview data.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views to fit the model on
            and cluster.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            Returns the predicted cluster labels for each sample.
        '''
        return
