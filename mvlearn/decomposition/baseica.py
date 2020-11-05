from .base import BaseDecomposer
import numpy as np
from fastsrm.identifiable_srm import IdentifiableFastSRM
from nonlinearsrm.grouppca import GroupPCA
from nonlinearsrm.groupica import GroupICA
from ..preprocessing.repeat import ViewTransformer
from sklearn.decomposition import PCA


class BaseICA(BaseDecomposer):
    """
    A base class for multiview ICA methods.
    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset.

    preproc: None, 'pca' or a ViewTransformer-like instance,
        default='pca'
        Preprocessing method to use to reduce data.
        If None, no preprocessing is applied.
        If "pca", performs PCA separately on each view to reduce dimension
        of each view.
        Otherwise the preprocessing is performed using the transform
        method of the ViewTransformer-like object ignoring the `n_components`
        parameter. This instance also needs an inverse transform method
        to recover original data from reduced data.

    multiview_output : bool, optional (default True)
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    random_state : int, RandomState instance or None, default=None
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.

    verbose : bool, default=False
        Print information

    Attributes
    ----------
    preproc : ViewTransformer-like instance
        The fitted instance used for preprocessing
    W_list : np array of shape (n_groups, n_components, n_components)
        The unmixing matrices to apply on preprocessed data
    """

    def __init__(
        self,
        n_components=None,
        preproc="pca",
        random_state=0,
        verbose=False,
        multiview_output=True,
    ):
        self.verbose = verbose
        self.n_components = n_components
        if preproc is None:
            self.preproc = None
        elif preproc == "pca":
            self.preproc = ViewTransformer(PCA(n_components=n_components))
        else:
            if hasattr(preproc, "transform") and hasattr(
                preproc, "inverse_transform"
            ):
                self.preproc = preproc
            else:
                raise ValueError(
                    "The dimension reduction instance needs"
                    "a transform and inverse transform method"
                )
        self.random_state = random_state
        self.multiview_output = multiview_output

    def fit(self, X, y=None):
        """
        Fits the model
        Parameters
        ----------
        X: list of np arrays of shape (n_voxels, n_samples)
            Input data: X[i] is the data of subject i
        """
        if self.preproc is not None:
            reduced_X = self.preproc.fit_transform(X)
        else:
            reduced_X = X.copy()
        reduced_X = np.array(reduced_X)
        W_list, _ = self.fit_(reduced_X.T)
        self.W_list = W_list
        return self

    def transform(self, X):
        r"""
        Recover the sources from each view (apply unmixing matrix).

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Training data to recover a source and unmixing matrices from.

        y : ignored

        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """
        if not hasattr(self, "W_list"):
            raise ValueError("The model has not yet been fitted.")

        if self.preproc is not None:
            transformed_X = self.preproc.transform(X)
        else:
            transformed_X = X.copy()
        if self.multiview_output:
            return [w.dot(x.T).T for w, x in zip(self.W_list, transformed_X)]
        else:
            return np.mean(
                [w.dot(x.T).T for w, x in zip(self.W_list, transformed_X)],
                axis=0,
            )

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, S):
        r"""
        Transforms the sources back to the mixed data for each view
        (apply mixing matrix).

        Parameters
        ----------
        None

        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """
        if not hasattr(self, "W_list"):
            raise ValueError("The model has not yet been fitted.")

        if self.multiview_output:
            S_ = np.mean(S, axis=0)
        else:
            S_ = S
        inv_red = [np.linalg.inv(w).dot(S_.T).T for w in self.W_list]

        if self.preproc is not None:
            return self.preproc.inverse_transform(inv_red)
        else:
            return inv_red
