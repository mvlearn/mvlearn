from sklearn.base import clone, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils import check_Xs


class Filter(TransformerMixin):
    r"""Apply a sklearn transformer to each view of a dataset

    Build a transformer from multiview dataset to multiview dataset by
    using individual scikit-learn transformer on each view.

    Parameters
    ----------
    transformer : a sklearn transformer instance, or a list
        Either a single sklearn transformer that will be applied to each
        view. One clone of the estimator will correspond to each view.
        Otherwise, it should be a list of estimators, of length the number of
        views in the multiview dataset.

    Attributes
    ----------
    n_views_ : int
        The number of views in the input dataset

    transformer_list_ : list of objects of length n_views_
        The list of transformer used to transform data. If
        self.transformer is a single transformer, it is a list containing
        clones of that transformer, otherwise it is a view of self.transformer.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.filter import Filter
    >>> from sklearn.decomposition import PCA
    >>> Xs, _ = load_UCImultifeature()
    >>> filt = Filter(PCA(n_components=2))
    >>> X_transformed = filt.fit_transform(Xs)
    >>> print(len(X_transformed))
    6
    >>> print(X_transformed[0].shape)
    (2000, 2)
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def prefit(self, Xs, y=None):
        r"""Estimate the attributes of the class.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Xs = check_Xs(Xs)
        self.n_views_ = len(Xs)
        if type(self.transformer) is list:
            if len(self.transformer) != self.n_views_:
                raise ValueError(
                    "The length of the transformers should be the same as the"
                    "number of views"
                )
            self.transformer_list_ = self.transformer
        else:
            self.transformer_list_ = [
                clone(self.transformer) for _ in range(self.n_views_)
            ]
        return self

    def fit(self, Xs, y=None):
        r"""Fit each estimator to the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.prefit(Xs, y)
        for transformer, X in zip(self.transformer_list_, Xs):
            transformer.fit(X)
        return self

    def fit_transform(self, Xs, y=None):
        r"""Fit and transform each dataset

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : None
            Ignored variable.

        Returns
        -------
        X_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        self.prefit(Xs, y)
        X_transformed = []
        for transformer, X in zip(self.transformer_list_, Xs):
            X_transformed.append(transformer.fit_transform(X))
        return X_transformed

    def transform(self, Xs, y=None):
        r"""Transform each dataset

        Applies the transform of each transformer on the
        individual views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        y : None
            Ignored variable.

        Returns
        -------
        X_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        X_transformed = []
        for transformer, X in zip(self.transformer_list_, Xs):
            X_transformed.append(transformer.transform(X))
        return X_transformed

    def inverse_transform(self, Xs, y=None):
        r"""Compute the inverse transform of a dataset

        Applies the inverse_transform function of each
        transformer on the individual datasets

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        y : None
            Ignored variable.

        Returns
        -------
        X_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs)
        for transformer in self.transformer_list_:
            if not hasattr(transformer, "inverse_transform"):
                raise AttributeError(
                    "A transfomer does no implement an inverse_transform "
                    "method"
                )
        X_transformed = []
        for transformer, X in zip(self.transformer_list_, Xs):
            X_transformed.append(transformer.inverse_transform(X))
        return X_transformed
