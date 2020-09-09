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
    Then an ICA is performed yielding the output dataset S. The unmixing matrix
    corresponding to data X are obtained by solving
    argmin_{W} ||S - WXs[i]||^2.

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

    max_iter : int
        Maximal number of iterations for the algorithm

    tol : float
        tolerance for the stopping criterion. Iterations stop when the norm
        of the gradient gets smaller than tol.

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
    components_ : array, shape (n_components, n_total_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``. `n_total_features` is the sum of all
        the number of input features.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.

    mean_ : array, shape (n_total_features, )
        Per-feature empirical mean, estimated from the training set.

    individual_components_ : list of array
        Individual components for each individual PCA.
        `individual_components_[i]` is an array of shape
        (n_individual_components, n_features) where n_features is the number of
        features in the dataset `i`.

    individual_explained_variance_ : list of array
        Individual explained variance for each individual PCA.
        `individual_explained_variance_[i]` is an array of shape
        (n_individual_components, ).

    individual_explained_variance_ratio_ : list of array
        Individual explained variance ratio for each individual PCA.
        `individual_explained_variance_ratio_[i]` is an array of shape
        (n_individual_components, ) where n_features is the number of
        features in the dataset `i`.

    individual_mean_ : list of array
        Individual mean for each individual PCA.
        `individual_mean_[i]` is an array of shape
        (n_features) where n_features is the number of
        features in the dataset `i`.

    n_components_ : int
        The estimated number of components.

    n_features_ : list of int
        Number of features in each training dataset.

    n_samples_ : int
        Number of samples in the training data.

    n_subjects_ : int
        Number of subjects in the training data

    unmixing_: np array of shape n_components, n_components
        Unmixing matrix. Sources are given by unmixing.dot(X_reduced)
        where X_reduced are the data reduced by GroupPCA

    mixings_: list of np array of shape n_components, n_features
        Subject specific mixing matrices
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
        self.whiten = whiten
        self.n_individual_components = n_individual_components
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
        gpca = GroupPCA(
            n_components=self.n_components,
            n_individual_components=self.n_individual_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        X = gpca.fit_transform(Xs)
        K, W, output = picard(X, max_iter=self.max_iter, tol=self.tol)
        self.unmixing_ = W.dot(K)
        self.mean_ = gpca.mean_
        self.individual_components_ = gpca.individual_components_
        self.individual_explained_variance_ = (
            gpca.individual_explained_variance_
        )
        self.individual_explained_variance_ratio_ = (
            gpca.individual_explained_variance_ratio_
        )
        self.individual_mean_ = gpca.individual_mean_
        self.n_components_ = gpca.n_components_
        self.n_features_ = gpca.n_features_
        self.n_samples_ = gpca.n_samples_
        self.n_subjects_ = gpca.n_subjects_
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
        gpca = GroupPCA(
            n_components=self.n_components,
            n_individual_components=self.n_individual_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        gpca.mean_ = self.mean_
        gpca.individual_components_ = self.individual_components_
        gpca.individual_explained_variance_ = (
            self.individual_explained_variance_
        )
        gpca.individual_explained_variance_ratio_ = (
            self.individual_explained_variance_ratio_
        )
        gpca.individual_mean_ = self.individual_mean_
        gpca.n_components_ = self.n_components_
        gpca.n_features_ = self.n_features_
        gpca.n_samples_ = self.n_samples_
        gpca.n_subjects_ = self.n_subjects_
        X = self.gpca.transform(Xs)
        return self.unmixing_.dot(X)

    def inverse_transform(self, X_transformed):
        r"""
        A method to recover multiview data from transformed data
        """
        gpca = GroupPCA(
            n_components=self.n_components,
            n_individual_components=self.n_individual_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        gpca.mean_ = self.mean_
        gpca.individual_components_ = self.individual_components_
        gpca.individual_explained_variance_ = (
            self.individual_explained_variance_
        )
        gpca.individual_explained_variance_ratio_ = (
            self.individual_explained_variance_ratio_
        )
        gpca.individual_mean_ = self.individual_mean_
        gpca.n_components_ = self.n_components_
        gpca.n_features_ = self.n_features_
        gpca.n_samples_ = self.n_samples_
        gpca.n_subjects_ = self.n_subjects_
        return self.gpca.inverse_transform(
            np.linalg.pinv(self.unmixing_).dot(X_transformed)
        )
