# License: MIT

from abc import abstractmethod

from sklearn.base import BaseEstimator


class BaseEmbed(BaseEstimator):
    """
    A base class for embedding multiview data.
    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs, y=None):
        """
        A method to fit model to multiview data.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)

        y : array, shape (n_samples,), optional

        Returns
        -------
        self: returns an instance of self.
        """

        return self

    @abstractmethod
    def transform(self, Xs):
        """
        Transform data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        Xs_transformed : list of array-likes
            - length: n_views
            - Xs_transformed[i] shape: (n_samples, n_components_i)
        """

        pass

    def fit_transform(self, Xs, y=None):
        """
        Fit an embedder to the data and transform the data

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
        y : array, shape (n_samples,), optional
            Targets to be used if fitting the algorithm is supervised.

        Returns
        -------
        X_transformed : list of array-likes
            - X_transformed length: n_views
            - X_transformed[i] shape: (n_samples, n_components_i)
        """
        return self.fit(Xs=Xs, y=y).transform(Xs=Xs)
