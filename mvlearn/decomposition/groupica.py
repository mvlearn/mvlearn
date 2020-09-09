import numpy as np
from picard import picard
from .grouppca import GroupPCA


from ..utils.utils import check_Xs
from .base import BaseEstimator


class GroupICA(BaseEstimator):
    r"""
    Group Independent component analysis.
    As an optional preprocessing, each dataset in `Xs` is reduced with
    usual PCA. Then, datasets are concatenated in the features direction,
    and a PCA is performed on this matrix, yielding a single dataset.
    Then an ICA is performed yielding the output dataset.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset

    n_individual_components : int or list of int or 'auto', optional
        The number of individual components to extract as a preprocessing.
        If None, no preprocessing is applied. If an `int`, each dataset
        is reduced to this dimension. If a list, the dataset `i` is
        reduced to the dimension `n_individual_components[i]`.
        If `'auto'`, set to the minimum between n_components and the
        smallest number of features in each dataset.

    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    random_state : int, RandomState instance, default=None
        Used when ``svd_solver`` == 'arpack' or 'randomized'. Pass an int
        for reproducible results across multiple function calls.

    Attributes
    ----------
    gpca_ : Instance of GroupPCA
        GroupPCA used to reduce the data

    unmixing: np array of shape n_components, n_components
        Unmixing matrix. Sources are given by unmixing.dot(X_reduced)
        where X_reduced are the data reduced by GroupPCA

    References
    ----------


    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import GroupICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = GroupICA(n_components=3)
    >>> Xs_transformed = ica.fit_transform(Xs)
    """

    def __init__(
        self,
        n_components=None,
        n_individual_components="auto",
        whiten=False,
        random_state=None,
        max_iter=100,
        tol=1e-7,
    ):
        self.gpca = GroupPCA(
            n_components=n_components,
            n_individual_components=n_individual_components,
            whiten=whiten,
            random_state=random_state,
        )
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, Xs, y=None):
        """
        Fit  to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            The transformed data
        """
        Xs = check_Xs(Xs)
        X = self.gpca.fit_transform(Xs)
        K, W, output = picard(X, max_iter=self.max_iter, tol=self.tol)
        self.unmixing = W.dot(K)
        return output

    def fit(self, Xs, y=None):
        r"""Fit the model with Xs.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(Xs, y)
        return self

    def transform(self, Xs, y=None):
        r"""
        A method to fit model to multiview data.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional

        Returns
        -------
        X_transformed : array of shape (n_samples, n_components)
            The transformed data
        """
        X = self.gpca.transform(Xs)
        return self.unmixing.dot(X)

    def inverse_transform(self, X_transformed):
        r"""
        A method to recover multiview data from transformed data
        """
        return self.gpca.inverse_transform(
            np.linalg.inv(self.unmixing).dot(X_transformed)
        )
