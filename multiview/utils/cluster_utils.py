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

import warnings
import numpy as np
from sklearn.utils import check_array


def check_Xs(Xs):

    '''
    Checks that the input Xs has the correct format.
    Parameters
    ----------
    Xs : list of array_likes
        - Xs shape (2,)
        - Xs[0] shape (n_samples, n_features_i)
        The data from two views.


    Returns
    -------
    Xs_arrays : list of numpy ndarrays
        The data as a list of numpy ndarrays

    '''

    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = "If not list, input must be of type np.ndarray"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    if len(Xs) != 2:
        msg = 'Xs must have 2 views'
        raise ValueError(msg)

    Xs_arrays = [check_array(X, allow_nd=False) for X in Xs]

    if (Xs_arrays[0].shape[0] != Xs_arrays[1].shape[0]):
        msg = 'Number of samples in both views must match'
        raise ValueError(msg)

    return Xs_arrays
