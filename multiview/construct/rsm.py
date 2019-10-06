"""
rsm.py
====================================
Random subspace method for view construction.
"""

import numpy as np
import random

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

        - If int, then consider n_features as number of columns to select.

        - If float, then consider n_features*n_cols as number of columns to select.

    n_views : int, float optional (default = 1)
        Number of views to construct.

    Returns
    -------
    views : list of array-like matrices
        List of constructed views (each matrix has shape [n_rows, n_cols]).

    """
     
    # todo let max_features be float for probability
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
        
        for feature in range(cols):
            if feature not in features_selected:
                view[:, feature] = 0
        views.append(view)
    
    return views


def check_n_views(n_views):
    """
    Checks to make sure n_views is a valid parameter.
    The following checks are made:
    * n_views must be greater than 1.
    * n_views must be an integer.
    The function raises an Exception if a check fails.

    Parameters
    ----------
    n_views: int
        Number of views to construct.
    """

    if n_views < 1:
        raise Exception("n_views must be >= 1.")

    if not isinstance(n_views, int):
        raise Exception("n_views must be an integer.")

    return


def check_n_features(n_features, cols):
    """
    Checks to make sure n_features is a valid parameter.
    The following checks are made:
    * n_features must be postive.
    * n_features must be integer if > 1.
    * n_features must be less than number of columns in data.
    The function raises an Exception if a check fails.

    Parameters
    ----------
    n_features: float, int
        Number of features to randomly select.
    
    cols: int
        Number of features in data matrix.
    """

    if n_features <= 0:
        raise Exception("n_features cannot be 0 or a negative number.")

    if n_features > 1 and not isinstance(n_features, int):
        raise Exception("n_features must be integer if > 1.")

    if n_features > cols:
        raise Exception("n_features must be less than columns in X.")

    return



