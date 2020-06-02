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

import numpy as np
from sklearn.base import BaseEstimator


class BaseFactorize(BaseEstimator):
    """
    A base class for factorizing multiview data.
    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs, y=None):
        r"""
        A method to fit model to multiview data.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self
