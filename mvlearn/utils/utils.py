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

import numpy as np
from sklearn.utils import check_X_y, check_array


def check_Xs(
    Xs,
    multiview=False,
    enforce_views=None,
    copy=False,
    return_dimensions=False,
):
    r"""
    Checks Xs and ensures it to be a list of 2D matrices.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, (default=False)
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs,
        and operations on the output will not affect
        the input.
        If False, the returned Xs is a view of the input Xs,
        and operations on the output will change the input.

    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of views and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each view.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    n_views : int, returned only if return_dimensions is True
        The number of views in the dataset.

    n_samples : int, returned only if return_dimensions is True
        The number of samples in the dataset.

    n_views : list, returned only if return_dimensions is True
        List of lenght n_views containing the number of features for each
        view.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_views = len(Xs)
    if n_views == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if n_views == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and n_views != enforce_views:
            msg = "Wrong number of views. Expected {} but found {}".format(
                enforce_views, n_views
            )
            raise ValueError(msg)

    Xs_converted = [check_array(X, allow_nd=False, copy=copy) for X in Xs]

    if not len(set([X.shape[0] for X in Xs_converted])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    if not return_dimensions:
        return Xs_converted
    else:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs_converted, n_views, n_samples, n_features


def check_Xs_y(
    Xs, y, multiview=False, enforce_views=None, return_dimensions=False
):
    r"""
    Checks Xs and y for consistent length. Xs is set to be of dimension 3.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    y : nd-array, list
        Labels.

    multiview : boolean, (default=False)
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of views and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each view.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    y_converted : object
        The converted and validated y.

    n_views : int
        The number of views in the dataset. Returned only if ``return_dimensions``
        is ``True``.

    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_features : list
        List of length ``n_views`` containing the number of features in
        each view. Returned only if ``return_dimensions`` is ``True``.
    """
    if return_dimensions:
        Xs_converted, n_views, n_samples, n_features = check_Xs(
            Xs,
            multiview=multiview,
            enforce_views=enforce_views,
            return_dimensions=True,
        )
    else:
        Xs_converted = check_Xs(
            Xs, multiview=multiview, enforce_views=enforce_views
        )
    _, y_converted = check_X_y(Xs_converted[0], y, allow_nd=False)

    if return_dimensions:
        return Xs_converted, y_converted, n_views, n_samples, n_features
    else:
        return Xs_converted, y_converted


def check_Xs_y_nan_allowed(
    Xs,
    y,
    multiview=False,
    enforce_views=None,
    num_classes=None,
    max_classes=None,
    min_classes=None
):
    r"""
    Checks Xs and y for consistent length. Xs is set to be of dimension 3.
    The labels (y) are allowed to be np.nan.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    y : nd-array, list
        Labels.

    multiview : boolean, default=False
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    num_classes : int, default=None
        Number of classes that must appear in the labels. If none, then
        not checked.

    max_classes : int, default=None
        Maximum number of classes that must appear in labels. If none, then
        not checked.

    min_classes : int, default=None
        Minimum number of classes that must appear in labels. If none, then
        not checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    y_converted : object
        The converted and validated y.
    """

    Xs_converted = check_Xs(
        Xs, multiview=multiview, enforce_views=enforce_views
    )

    y_converted = np.array(y)
    if len(y_converted) != Xs_converted[0].shape[0]:
        msg = (
            "Incompatible label length {} for "
            " data with {} samples".format(
                len(y_converted), Xs_converted[0].shape[0]
            )
        )
        raise ValueError(msg)

    if num_classes is not None:
        # if not exactly correct number of class labels, raise error
        classes = list(set(y[~np.isnan(y)]))
        n_classes = len(classes)
        if n_classes != num_classes:
            raise ValueError(
                "Wrong number of class labels. Expected {},\
             found {}".format(
                    num_classes, n_classes
                )
            )
    if max_classes is not None:
        # if not exactly correct number of class labels, raise error
        classes = list(set(y[~np.isnan(y)]))
        n_classes = len(classes)
        if n_classes > max_classes:
            raise ValueError(
                "Wrong number of class labels. Expected no\
             more than {}, found {}".format(
                    num_classes, n_classes
                )
            )

    return Xs_converted, y_converted
