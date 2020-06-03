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

from abc import abstractmethod
from sklearn.base import BaseEstimator

class BaseCluster(BaseEstimator):
	'''
    A base class for clustering.
    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    '''
    def __init__():
    	pass

    @abstractmethod
    def fit_predict(self, Xs):

        '''
        A method for fitting then predicting cluster assignments.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        labels : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.
        '''

        pass
