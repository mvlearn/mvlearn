"""Multiview Random Gaussian Projection"""

# Authors: Ronan Perry
#
# License: MIT


import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.random_projection import GaussianRandomProjection
from .utils import check_n_views


class RandomGaussianProjection(TransformerMixin):
    """
    Random Gaussian Projection method for constructing multiple views.
    Each view is constructed using sklearn's random Gaussian projection.

    Parameters
    ----------
    n_views : int
        Number of views to construct

    n_components: int or 'auto', optional (default "auto")
        Dimensionality of target projection space, see
        sklearn.random_projection.GaussianRandomProjection for details.

    eps: float, optional (default 0.1)
        Parameter for controlling quality of embedding when
        n_components = "auto" according to the Johnson-Lindenstrauss lemma
        A smaller value leads to a better emedding (see sklearn for details).

    random_state : int or RandomState instance, optional (default None)
        Controls the random sampling of Gaussian projections. Set for
        reproducible results.

    Returns
    -------
    Xs : list of array-likes matrices
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_components)

    Attributes
    ----------
    GaussianRandomProjections_  : list, length n_views
        List of GaussianRandomProjection instances fitted to construct each
        view.

    Notes
    -----
    From an implementation perspective, this wraps GaussianRandomProjection
    from `sklearn.random_projection <https://scikit-learn.org/stable/modules/
    classes.html#module-sklearn.random_projection>`_ and creates multiple
    projections.

    Examples
    --------
    >>> from mvlearn.compose import RandomGaussianProjection
    >>> import numpy as np
    >>> X = np.random.rand(1000, 50)
    >>> rgp = RandomGaussianProjection(n_views=3, n_components=10)
    >>> Xs = rgp.fit_transform(X)
    >>> print(len(Xs))
    3
    >>> print(Xs[0].shape)
    (1000, 10)
    """
    def __init__(self, n_views, n_components="auto", eps=0.1,
                 random_state=None):
        check_n_views(n_views)
        self.n_views = n_views
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    def fit(self, X, y=None):
        r"""
        Fit to the singleview data.

        Parameters
        ----------
        X : array of shape (n_samples, n_total_features)
            Input dataset

        y : Ignored

        Returns
        -------
        self : object
            The Transformer instance
        """
        # set function level random state
        np.random.seed(self.random_state)
        self.GaussianRandomProjections_ = [
            GaussianRandomProjection(
                n_components=self.n_components, eps=self.eps).fit(X)
            for _ in range(self.n_views)
        ]

        return self

    def transform(self, X):
        r"""
        Transforms the singleview dataset and into a multiview dataset.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input dataset

        Returns
        -------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_components)
        """
        check_is_fitted(self)
        Xs = [grp.transform(X) for grp in self.GaussianRandomProjections_]
        return Xs
