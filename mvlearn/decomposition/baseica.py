# MIT License

# Copyright (c) [2017] [Pierre Ablin and Hugo Richard]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from .base import BaseDecomposer
from ..preprocessing.repeat import ViewTransformer


class BaseICA(BaseDecomposer):
    """A base class for multiview ICA methods.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset.

    preproc: None, 'pca' or a ViewTransformer-like instance,\
            default='pca'
        Preprocessing method to use to reduce data.
        If None, no preprocessing is applied.
        If "pca", performs PCA separately on each view to reduce dimension
        of each view.
        Otherwise the preprocessing is performed using the transform
        method of the ViewTransformer-like object ignoring the `n_components`
        parameter. This instance also needs an inverse transform method
        to recover original data from reduced data.

    multiview_output : bool, default True
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
    preproc_instance : ViewTransformer-like instance
        The fitted instance used for preprocessing

    mixing_ : array, shape (n_views, n_components, n_components)
        The square mixing matrices, linking preprocessed data
        and the independent components.

    pca_components_: array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions
        of maximum variance in the data. Only used if preproc == "pca".

    components_ : array, shape (n_views, n_components, n_components)
        The square unmixing matrices

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
        self.preproc = preproc
        if preproc is None or preproc == "pca" and n_components is None:
            self.preproc_instance = None
            self.preproc_name = "None"
        elif preproc == "pca":
            self.preproc_name = "pca"
            self.preproc_instance = ViewTransformer(
                PCA(n_components=n_components)
            )
        else:
            if hasattr(preproc, "transform") and hasattr(
                preproc, "inverse_transform"
            ):
                self.preproc_name = "custom"
                self.preproc_instance = preproc
            else:
                raise ValueError(
                    "The dimension reduction instance needs"
                    "a transform and inverse transform method"
                )
        self.random_state = random_state
        self.multiview_output = multiview_output

    def fit(self, Xs, y=None):
        """Fits the model.

        Parameters
        ----------
        Xs: list of np arrays of shape (n_voxels, n_samples)
            Input data: X[i] is the data of subject i

        y : ignored
        """
        if self.preproc_instance is not None:
            reduced_X = self.preproc_instance.fit_transform(Xs)
        else:
            reduced_X = Xs.copy()
        reduced_X = np.array(reduced_X)
        unmixings_, S = self._fit(np.swapaxes(reduced_X, 1, 2))
        mixing_ = np.array([np.linalg.pinv(W) for W in unmixings_])
        self.components_ = unmixings_
        self.mixing_ = mixing_
        if self.preproc_name == "pca":
            pca_components = []
            for i, transformer in enumerate(
                self.preproc_instance.transformers_
            ):
                K = transformer.components_
                pca_components.append(K)
            self.pca_components_ = np.array(pca_components)

        if self.preproc_name == "None":
            self.individual_components_ = unmixings_
            self.individual_mixing_ = mixing_
        else:
            self.individual_mixing_ = []
            self.individual_components_ = []
            sources_pinv = linalg.pinv(S.T)
            for x in Xs:
                lstq_solution = np.dot(sources_pinv, x)
                self.individual_components_.append(
                    linalg.pinv(lstq_solution).T
                )
                self.individual_mixing_.append(lstq_solution.T)
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

        Returns
        -------
        Xs_new : numpy.ndarray, shape (n_views, n_samples, n_components)
            The mixed sources from the single source and per-view unmixings.
        """
        if not hasattr(self, "components_"):
            raise ValueError("The model has not yet been fitted.")

        if self.preproc_instance is not None:
            transformed_X = self.preproc_instance.transform(X)
        else:
            transformed_X = X.copy()
        if self.multiview_output:
            return np.array(
                [x.dot(w.T) for w, x in zip(self.components_, transformed_X)]
            )
        else:
            return np.mean(
                [x.dot(w.T) for w, x in zip(self.components_, transformed_X)],
                axis=0,
            )

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

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
        if not hasattr(self, "components_"):
            raise ValueError("The model has not yet been fitted.")

        if self.multiview_output:
            S_ = np.mean(X_transformed, axis=0)
        else:
            S_ = X_transformed
        inv_red = [S_.dot(w.T) for w in self.mixing_]

        if self.preproc_instance is not None:
            return self.preproc_instance.inverse_transform(inv_red)
        else:
            return inv_red
