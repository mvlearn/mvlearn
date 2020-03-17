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


def check_Xs(Xs, multiview=False, enforce_views=None):
    """
    Checks Xs and ensures it to be a list of 2D matrices.
    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, default (False)
        Throws error if just 1 data matrix

    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = "If not list, input must be of type np.ndarray"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    if len(Xs) == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if len(Xs) == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and len(Xs) != enforce_views:
            msg = "Wrong number of views. Expected {enforce_views} " \
                " but found {len(Xs)}"
            raise ValueError(msg)

    Xs_converted = [check_array(X, allow_nd=False) for X in Xs]

    if not len(set([X.shape[0] for X in Xs_converted])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    return Xs_converted


def check_Xs_y(Xs, y, multiview=False, enforce_views=None):
    """
    Checks Xs and y for consistent length. Xs is set to be of dimension 3
    Parameters
    ----------
    Xs : nd-array, list
        Input data. F

    y : nd-array, list
        Labels.

    multiview : boolean, default (False)
        Throws error if just 1 data matrix.

    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.
    Returns
    -------
    Xs_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    Xs_converted = check_Xs(Xs, multiview=multiview,
                            enforce_views=enforce_views)
    _, y_converted = check_X_y(Xs_converted[0], y, allow_nd=False)

    return Xs_converted, y_converted


def check_Xs_y_nan_allowed(
        Xs,
        y,
        multiview=False,
        enforce_views=None,
        num_classes=None
        ):
    """
    Checks Xs and y for consistent length. Xs is set to be of dimension 3
    Parameters
    ----------
    Xs : nd-array, list
        Input data.
    y : nd-array, list
        Labels.
    multiview : boolean, default (False)
        Throws error if just 1 data matrix
    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.
    num_classes: int, default (None)
        Number of classes that must appear in the labels. If none, then
        not checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """

    Xs_converted = check_Xs(Xs,
                            multiview=multiview,
                            enforce_views=enforce_views)

    y_converted = np.array(y)
    if len(y_converted) != Xs_converted[0].shape[0]:
        raise ValueError("Incompatible label length {len(y_converted)} for "
                         " data with {Xs_converted[0].shape[0]} samples")

    if num_classes is not None:
        # if not exactly correct number of class labels, raise error
        classes = list(set(y[~np.isnan(y)]))
        n_classes = len(classes)
        if n_classes != num_classes:
            raise ValueError("Wrong number of class labels. Expected {},\
             found {}".format(num_classes, n_classes))

    return Xs_converted, y_converted
