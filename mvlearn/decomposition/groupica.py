"""Group Independent Component Analysis."""

# Authors: Pierre Ablin, Hugo Richard
#
# License: MIT

import numpy as np
from scipy import linalg
from sklearn.decomposition import fastica
from sklearn.utils.validation import check_is_fitted

try:
    from picard import picard
except ModuleNotFoundError as error:
    msg = (f"ModuleNotFoundError: {error}. multiviewica dependencies" +
           "required for this function. Please consult the mvlearn" +
           "installation instructions at https://github.com/mvlearn/mvlearn" +
           "to correctly install multiviewica dependencies.")
    raise ModuleNotFoundError(msg)

from ..utils.utils import check_Xs
from .base import BaseDecomposer
from .grouppca import GroupPCA


class GroupICA(BaseDecomposer):
    r"""Group Independent component analysis.

    Each dataset in `Xs` is reduced with usual PCA (this step is optional).
    Then, datasets are concatenated in the features direction, and a PCA is
    performed on this matrix, yielding a single dataset. ICA is finally
    performed yielding the output dataset S. The unmixing matrix W
    corresponding to data X are obtained by solving
    argmin_{W} ||S - WX||^2.

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

    multiview_output : bool, optional (default True)
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    prewhiten : bool, optional (default False)
        Whether the data should be whitened after the original preprocessing.

    solver : str {'picard', 'fastica'}
        Chooses which ICA solver to use. `picard` is generally faster and
        more reliable.

    ica_kwargs : dict
        Optional keyword arguments for the ICA solver. If solver='fastica',
        see the documentation of sklearn.decomposition.fastica.
        If solver='picard', see the documentation of picard.picard.

    random_state : int, RandomState instance, default=None
        Controls the random number generator used in the estimator. Pass an int
        for reproducible output across multiple function calls.

    Attributes
    ----------
    means_ : list of arrays of shape (n_components,)
        The mean of each dataset

    grouppca_ : mvlearn.decomposition.GroupPCA instance
        A GroupPCA class for preprocessing and dimension reduction

    mixing_ : array, shape (n_components, n_components)
        The square mixing matrix, linking the output of the group-pca
        and the independent components.

    components_ : array, shape (n_components, n_components)
        The square unmixing matrix

    individual_components_ : list of array
        Individual unmixing matrices estimated by least squares.
        `individual_components_[i]` is an array of shape
        (n_components, n_features) where n_features is the number of
        features in the dataset `i`.

    individual_mixing_ : list of array
        Individual mixing matrices estimated by least squares.
        `individual_components_[i]` is an array of shape
        (n_features, n_components) where n_features is the number of
        features in the dataset `i`.

    n_components_ : int
        The estimated number of components.

    n_features_ : list of int
        Number of features in each training dataset.

    n_samples_ : int
        Number of samples in the training data.

    n_views_ : int
        Number of views in the training data

    See also
    --------
    GroupPCA
    multiviewica

    References
    ----------
    .. [#1groupica] Vince D Calhoun, et al.
            "A method for making group inferences from
            functional MRI data using independent component analysis."
            Human Brain Mapping, 14(3):140–151, 2001.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import GroupICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = GroupICA(n_components=3)
    >>> Xs_transformed = ica.fit_transform(Xs)
    >>> print(len(Xs_transformed))
    6
    >>> print(Xs_transformed[0].shape)
    (2000, 3)
    """

    def __init__(
        self,
        n_components=None,
        n_individual_components="auto",
        multiview_output=True,
        prewhiten=False,
        solver="picard",
        ica_kwargs={},
        random_state=None,
    ):
        if solver not in ["picard", "fastica"]:
            raise ValueError(
                "Invalid solver, must be either `fastica` or `picard`"
            )
        self.n_components = n_components
        self.n_individual_components = n_individual_components
        self.multiview_output = multiview_output
        self.prewhiten = prewhiten
        self.solver = solver
        self.ica_kwargs = ica_kwargs
        self.random_state = random_state

    def fit(self, Xs, y=None):
        r"""Fit to the data.

        Estimate the parameters of the model

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        y : None
            Ignored variable.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        Xs = check_Xs(Xs, copy=True)
        self.means_ = [np.mean(X, axis=0) for X in Xs]
        gpca = GroupPCA(
            n_components=self.n_components,
            n_individual_components=self.n_individual_components,
            prewhiten=self.prewhiten,
            whiten=True,
            multiview_output=False,
            random_state=self.random_state,
        )
        X_pca = gpca.fit_transform(Xs)
        self.grouppca_ = gpca
        if self.solver == "fastica":
            K, W, sources = fastica(
                X_pca, **self.ica_kwargs, random_state=self.random_state
            )
        else:
            K, W, sources = picard(
                X_pca.T, **self.ica_kwargs, random_state=self.random_state
            )
            sources = sources.T
        if K is not None:
            self.components_ = np.dot(W, K)
        else:
            self.components_ = W
        self.mixing_ = linalg.pinv(self.components_)
        # Compute individual unmixing matrices by least-squares
        self.individual_mixing_ = []
        self.individual_components_ = []
        sources_pinv = linalg.pinv(sources)
        for X, mean in zip(Xs, self.means_):
            lstq_solution = np.dot(sources_pinv, X - mean)
            self.individual_components_.append(linalg.pinv(lstq_solution).T)
            self.individual_mixing_.append(lstq_solution.T)
        self.n_components_ = gpca.n_components_
        self.n_features_ = gpca.n_features_
        self.n_samples_ = gpca.n_samples_
        self.n_views_ = gpca.n_views_
        return self

    def transform(self, Xs, y=None, index=None):
        r"""Transform the data Xs into sources.

        Parameters
        ----------
        Xs: list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)

        y : None
            Ignored variable.

        index: None, or int or array-like
            int or list of ints specifying the indices of the
            inputted views relative to the fitted views.
            If None, there should be as many inputted views as fitted views.
            Note that the index parameter is not available in all methods of
            mvlearn yet.

        Returns
        -------
        X_transformed : list of array-likes or numpy.ndarray
            The transformed data.
            If `multiview_output` is True, it is a list with the estimated
            individual sources.
            If `multiview_output` is False, it is a single array containing the
            shared sources.
        """
        check_is_fitted(self)
        Xs = check_Xs(Xs, copy=True)
        if index is None:
            index_ = np.arange(self.n_views_)
        else:
            index_ = np.copy(index)

        assert len(index_) == len(Xs)

        if self.multiview_output:
            return [
                np.dot(X - mean, W.T)
                for W, X, mean in (
                    zip(
                        [self.individual_components_[i] for i in index_],
                        Xs,
                        [self.means_[i] for i in index_],
                    )
                )
            ]
        else:
            X = self.grouppca_.transform(Xs, index=index)
            return np.dot(X, self.components_.T)

    def inverse_transform(self, X_transformed, index=None):
        r"""Recover multiview data from transformed data.

        Parameters
        ----------
        X_transformed : list of array-likes or numpy.ndarray
            If `multiview_output` is True, it must be a list of arrays of shape
            (n_samples, n_components) containing estimated sources.
            If `multiview_output` is False, it must be a single
            array containing shared sources.

        index: None, or int or array-like
            int or list of ints specifying the indices of the
            inputted views relative to the fitted views.
            If None, there should be as many inputted views as fitted views.
            Note that the index parameter is not available in all methods of
            mvlearn yet.


        Returns
        -------
        Xs : list of arrays
            The recovered individual datasets.
        """
        check_is_fitted(self)

        if index is None:
            index_ = np.arange(self.n_views_)
        else:
            index_ = np.copy(index)

        if self.multiview_output:
            X_transformed = check_Xs(X_transformed)
            assert len(X_transformed) == len(index_)
            return [
                np.dot(X, A.T) + mean
                for X, A, mean in (
                    zip(
                        X_transformed,
                        [self.individual_mixing_[i] for i in index_],
                        [self.means_[i] for i in index_],
                    )
                )
            ]

        else:
            return self.grouppca_.inverse_transform(
                np.dot(X_transformed, self.mixing_.T), index=index
            )
