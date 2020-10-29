# License: MIT

import numpy as np
import random
from .utils import check_n_views, check_n_features


def random_subspace_method(X, n_features=None, n_views=1):
    """
    Random Subspace Method [#1RSM]_ for constructing multiple views.
    Each view is constructed by randomly selecting n_features
    (columns) from X. All unselected features are set to 0.

    Parameters
    ----------
    X : array-like matrix, shape = (n_samples, n_cols)
        The input samples.

    n_features : int, float
        Number (or proportion, if float) of features to randomly select.
            - If int, then consider n_features as number of columns
              to select.

            - If float, then consider n_features*n_cols as number of columns
              to select.

    n_views : strictly positive int, float optional (default = 1)
        Number of views to construct.

    Returns
    -------
    views : list of array-like matrices
        List of constructed views.
            - length: n_views
            - each view has shape (n_samples, n_features)

    References
    ----------
    .. [#1RSM] Ho, Tin Kam. "The random subspace method for constructing
            decision forests." IEEE transactions on pattern analysis
            and machine intelligence 20.8 (1998): 832-844.

    Examples
    --------
    >>> from mvlearn.construct import random_subspace_method
    >>> import random
    >>> import numpy as np
    >>> # Random integer data for compressed viewing
    >>> np.random.seed(1)
    >>> random.seed(1)
    >>> single_view_data = np.random.randint(low=1, high=10, size=(4, 5))
    >>> multi_view_data = random_subspace_method(single_view_data,
    ...                                          n_features=3, n_views=2)
    >>> print(multi_view_data[0])
    [[6 9 0 0 1]
     [2 8 0 0 5]
     [6 3 0 0 5]
     [8 8 0 0 1]]
    >>> print(multi_view_data[1])
    [[6 0 6 1 0]
     [2 0 7 3 0]
     [6 0 5 3 0]
     [8 0 2 8 0]]
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
