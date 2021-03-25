# License: MIT

from abc import abstractmethod
from sklearn.utils.validation import check_is_fitted

from sklearn.base import BaseEstimator
import numpy as np
from scipy import linalg

from ..compose import ViewTransformer
from sklearn.decomposition import PCA


class BaseDecomposer(BaseEstimator):
    """
    A base class for decomposing multiview data.
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
        self: returns an instance of self.
        """

        return self

    @abstractmethod
    def transform(self, Xs):
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
        Xs_transformed : list of array-likes
            - length: n_views
            - Xs_transformed[i] shape: (n_samples, n_components_i)
        """

        pass

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
        X_transformed : list of array-likes
            - out length: n_views
            - out[i] shape: (n_samples, n_components_i)
        """
        return self.fit(Xs, y).transform(Xs)


class BaseMultiView(BaseDecomposer):
    r"""
    Base for Multiview-ICA like algorithms 
    """

    def __init__(
        self, n_components=None, multiview_output=True,
    ):
        self.n_components = n_components
        self.multiview_output = multiview_output

    @abstractmethod
    def fit_reduced(self, reduced_X):
        """
        Fit the model with reduced data
        Parameters
        ----------
        reduced_X: np array of shape (n_views, n_samples, n_components)

        Returns
        --------
        unmixing: np array of shape (n_views, n_components, n_components)
            such that reduced_X[i].dot(unmixing[i].T) unmixes the data
        sources: np array of shape (n_samples, n_components)
        """
        pass

    @abstractmethod
    def aggregate(self, X_transformed):
        """
        Aggregate transformed data to form a unique output

        Returns
        -------
        Source: np array of shape (n_samples, n_components)
        """
        pass

    def fit(self, Xs, y=None):
        """Fits the model.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
        y : ignored

        Returns
        -------
        self: returns an instance of self.
        """
        if self.n_components is not None:
            self.pcas_ = ViewTransformer(PCA(n_components=self.n_components))

        if self.n_components is not None:
            reduced_X = self.pcas_.fit_transform(Xs)
        else:
            reduced_X = Xs.copy()
        reduced_X = np.array(reduced_X)
        unmixings_, S = self.fit_reduced(reduced_X)
        mixing_ = np.array([np.linalg.pinv(W) for W in unmixings_])
        self.components_ = unmixings_
        self.mixing_ = mixing_
        if self.n_components is not None:
            pca_components = []
            for i, estimator in enumerate(self.pcas_.estimators_):
                K = estimator.components_
                pca_components.append(K)
            self.pca_components_ = np.array(pca_components)

        if self.n_components is None:
            self.individual_components_ = unmixings_
            self.individual_mixing_ = mixing_
        else:
            self.individual_mixing_ = []
            self.individual_components_ = []
            sources_pinv = linalg.pinv(S)
            for x in Xs:
                lstq_solution = np.dot(sources_pinv, x)
                self.individual_components_.append(
                    linalg.pinv(lstq_solution).T
                )
                self.individual_mixing_.append(lstq_solution.T)
        return self

    def transform(self, Xs):
        r"""
        Recover the sources from each view (apply unmixing matrix).

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Training data to recover a source and unmixing matrices from.

        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """

        if not hasattr(self, "components_"):
            raise ValueError("The model has not yet been fitted.")

        if self.n_components is not None:
            transformed_X = self.pcas_.transform(Xs)
        else:
            transformed_X = Xs.copy()
        if self.multiview_output:
            return np.array(
                [x.dot(w.T) for w, x in zip(self.components_, transformed_X)]
            )
        return self.aggregate(
            [x.dot(w.T) for w, x in zip(self.components_, transformed_X)],
        )

    def inverse_transform(self, X_transformed):
        r"""
        Transforms the sources back to the mixed data for each view
        (apply mixing matrix).

        Parameters
        ----------
        X_transformed : list of array-likes or numpy.ndarray
            The dataset corresponding to transformed data
        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """
        check_is_fitted(self, "components_")
        if self.multiview_output:
            S_ = self.aggregate(X_transformed)
        else:
            S_ = X_transformed
        inv_red = [S_.dot(w.T) for w in self.mixing_]

        if self.n_components is not None:
            return self.pcas_.inverse_transform(inv_red)
        else:
            return inv_red
