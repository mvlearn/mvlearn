"""
rsm.py
====================================
Random subspace method for view construction.
"""

import numpy as np
import random
from .utils import check_n_views, check_n_features


def random_subspace_method(X, n_features=None, n_views=1):
    """
    Random Subspace Method for constructing multiple views.
    Each view is constructed by randomly selecting n_features
    (columns) from X. All unselected features are set to 0.

    Original paper: https://ieeexplore.ieee.org/document/709601

    Parameters
    ----------
    X : array-like matrix, shape = [n_rows, n_cols]
        The input samples.

    n_features : int, float
        Number of features to randomly select.

        - If int, then consider n_features as number of columns
        to select.

        - If float, then consider n_features*n_cols as number of columns
        to select.

    n_views : strictly positive int, float optional (default = 1)
        Number of views to construct.

    Returns
    -------
    views : list of array-like matrices
        List of constructed views (each matrix has shape [n_rows, n_cols]).

    Examples
    --------
    >>> from mvlearn.construct import random_subspace_method
    >>> single_view_data = np.random.rand(1000,50)
    >>> single_view_data = np.random.rand(1000, 50)
    >>> # Randomly select 10 of the 50 features to keep nonzero in each view
    >>> multi_view_data = random_subspace_method(single_view_data,
                                                 n_features=10, n_views=3)
    >>> print(len(multi_view_data))
    '3'
    >>> print(multi_view_data[0].shape)
    '(1000, 50)'
    """
    _, cols = X.shape

    check_n_views(n_views)
    check_n_features(n_features, cols)

    # check if n_feaures is between 0 and 1
    if n_features < 1:
        n_features = int(n_features*cols)

    views = []

    for _ in range(n_views):
        view = np.copy(X)
        features_selected = []

        while len(features_selected) != n_features:

            feature = random.randint(0, cols - 1)

            if feature not in features_selected:
                features_selected.append(feature)

        # set unselected features to zero
        for feature in range(cols):
            if feature not in features_selected:
                view[:, feature] = 0
        views.append(view)

    return views
