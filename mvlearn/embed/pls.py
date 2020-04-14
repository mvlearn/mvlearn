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
# Implements the Partial Least Squares view construction method
# by wrapping the PLS regression function from sklearn. View
# construction generates a new projection of the data.

from sklearn.cross_decomposition import PLSRegression
import numpy as np


def partial_least_squares_embedding(
    X, Y, n_components=2, return_weights=False
):
    """
    Calculates the Partial Least Squares embedding of the data X given the
    target (covariates) Y using the NIPALS algorithm implemented by sklearn.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features) Training vectors, where
        n_samples is the number of samples and n_features is the number of
        predictors.
    Y : array-like, shape = (n_samples, n_targets)
        Target vectors, where n_samples is the number of samples and
        n_targets is the number of response variables.
    n_components : int, 2 (default)
        The dimension of the embedded space
    return_weights : boolean, False (default)
        whether to return the projection weights instead of embeddings

    Returns
    -------
    X_embedding : shape (n_samples, n_components), default
        The embedded X data using the PLS weights.
    OR (if return_embedding is False)
    X_weights : shape (features, n_components)
        The PLS feature weights to embed the data

    Notes
    -----
    From an implementation perspective, this wraps PLSRegresion from
    sklearn.cross_decomposition.
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)

    # Extract projection (score) weights for each feature
    W = pls.x_weights_

    # Extract the loadings, the regression coefficients of X onto scores
    P = pls.x_loadings_

    # Calculate the correct projection weights for calculating >1 scores,
    # accounts for the NIPALS deflation procedure
    R = W @ np.linalg.pinv(P.T @ W)
    if return_weights:
        return R
    else:
        return X @ R
