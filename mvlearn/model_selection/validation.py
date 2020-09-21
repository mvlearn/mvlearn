"""Validation utils."""
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
# Authors: Pierre Ablin
import numpy as np

from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline

from ..utils import check_Xs
from ..split import ConcatSplitter


def cross_validate(estimator, Xs, y, *args, **kwargs):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Works on multiview data, by wrapping
    `sklearn.model_selection.cross_validate`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    Xs_transformed : list of array-likes
        - Xs shape: (n_views,)
        - Xs[i] shape: (n_samples, n_features_i)
        The multiview data to fit

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    args : any
        Additional arguments passed to `sklearn.model_selection.cross_validate`

    kwargs : any
        Additional named arguments to `sklearn.model_selection.cross_validate`

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        The output of `sklearn.model_selection.cross_validate`.
    """
    Xs = check_Xs(Xs)
    n_features = [X.shape[1] for X in Xs]
    pipeline = Pipeline([('splitted', ConcatSplitter(n_features)),
                         ('estimator', estimator)])
    return sk_cross_validate(pipeline, np.hstack(Xs), y, *args, **kwargs)
