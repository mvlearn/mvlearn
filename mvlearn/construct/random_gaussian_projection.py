"""
random_gaussian_projection.py
====================================
Random gaussian projection for view construction.
"""

import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from mvlearn.construct.utils import check_n_views


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
    X : array-like matrix, shape = [n_rows, n_cols]
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
        List of constructed views (each matrix has shape [n_rows, n_cols]).
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
