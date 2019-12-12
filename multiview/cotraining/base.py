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
# This is a base class for implementing multi-view estimators using
# co-training.

from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator


class BaseCoTrainEstimator(BaseEstimator):
    """
    A base class for multiview co-training. Extends from sklearn's
    BaseEstimator.

    Parameters
    ----------
    estimator1 : estimator object, default (None)
        The estimator object which will be trained on view 1 of the data.

    estimator2 : estimator object, default (None)
        The estimator object which will be trained on view 2 of the data.
        Does not need to be of the same type as estimator1.

    random_state : int
        The starting random seed for fit() and class operations, passed to
        numpy.random.seed().

    Attributes
    ----------
    estimator1 : estimator object
        The estimator used on view 1.

    estimator2 : estimator object
        The estimator used on view 2.

    random_state : int
        The starting random seed for fit() and class operations, passed to
        numpy.random.seed().
    """

    def __init__(self,
                 estimator1=None,
                 estimator2=None,
                 random_state=0
                 ):
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.random_state = random_state

    @abstractmethod
    def fit(self, Xs, y):
        """
        A method to co-trained estimators to multiview data.

        Parameters
        ----------
        Xs: numpy.ndarray, list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,)

        Returns
        -------
        self : object
        """

        return self

    @abstractmethod
    def predict(self, Xs):
        """
        A method to predict the class of multiview data.

        Parameters
        ----------
        Xs: numpy.ndarray, list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        y : array-like (n_samples, n_classes)
        """

        return self
