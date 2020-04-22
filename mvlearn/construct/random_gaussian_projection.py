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
from sklearn.random_projection import GaussianRandomProjection
from .utils import check_n_views


def random_gaussian_projection(X, n_views=1, n_components="auto",
                               eps=0.1,
                               random_state=None):
    """
    Random Gaussian Projection method for constructing multiple views.
    Each view is constructed using sklearn's random Gaussian projection.
    This wrapped version has an option to specify the number of views
    you want to generate. Random_state is also only set once in the
    function (not per view).

    Parameters
    ----------
    X : array-like matrix, shape = (n_samples, n_cols)
        The input samples.

    n_views : int, float optional (default = 1)
        Number of views to construct.

    n_components: int, string optional (default = "auto")
        Dimensionality of target projection space (see sklearn for details)

    eps: strictly positive float, optional (default = 0.1)
        Parameter for controlling quality of embedding when
        n_components = "auto" (see sklearn for details)

    random_state: int or None (default = None)
        Sets random state using np.random.seed


    Returns
    -------
    views : list of array-like matrices
        List of constructed views.
            - length: n_views
            - each view has shape (n_samples, n_features)

    Notes
    -----
    From an implementation perspective, this wraps GaussianRandomProjection
    from `sklearn.random_projection <https://scikit-learn.org/stable/modules/
    classes.html#module-sklearn.random_projection>`_ and creates multiple
    projections.

    Examples
    --------
    >>> from mvlearn.construct import random_gaussian_projection
    >>> import numpy as np
    >>> single_view_data = np.random.rand(1000, 50)
    >>> # Project to 10 components
    >>> multi_view_data = random_gaussian_projection(single_view_data,
    ...                                              n_views=3,
    ...                                              n_components=10)
    >>> print(len(multi_view_data))
    3
    >>> print(multi_view_data[0].shape)
    (1000, 10)
    """

    check_n_views(n_views)
    views = []
    # set function level random state
    np.random.seed(random_state)

    for _ in range(n_views):
        transformer = GaussianRandomProjection(n_components=n_components,
                                               eps=eps)
        X_proj = transformer.fit_transform(X)
        views.append(X_proj)

    return views
