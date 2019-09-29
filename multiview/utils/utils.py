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

from sklearn.utils import check_X_y, check_array
import numpy as np


def check_Xs(Xs):
    """
    Checks Xs and sets it to be of dimension 3
    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    """
    Xs = check_array(Xs, allow_nd=True)
    if Xs.ndim > 3:
        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], -1))
    elif Xs.ndim == 2:
        Xs = Xs.reshape((1, Xs.shape[0], Xs.shape[1]))


def check_Xs_y(Xs, y):
    """
    Checks Xs and y for consistent length. Xs is set to be of dimension 3
    Parameters
    ----------
    Xs : nd-array, list
        Input data.
    y : nd-array, list
        Labels.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """
    Xs = check_array(Xs)
    Xs_converted, y_converted = check_X_y(np.swapaxes(Xs, 0, 1), y, allow_nd=True)
    Xs_converted = np.swapaxes(Xs_converted, 0, 1)

    return Xs_converted, y_converted
