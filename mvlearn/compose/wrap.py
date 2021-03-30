"""Singleview function wrapping utilities."""

# Authors: Pierre Ablin, Ronan Perry
#
# License: MIT

from scipy import stats
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from ..utils import check_Xs, check_Xs_y
import numpy as np


class BaseWrapper(BaseEstimator):
    """Wraps an sklearn-compliant estimator for use on multiple views"""
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def _prefit(self, Xs, y=None):
        r"""Estimate the attributes of the class.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : array-like of length (n_samples,), optional (default None)
            Targets for a supervised estimation task

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if y is None:
            Xs = check_Xs(Xs)
        else:
            Xs, y = check_Xs_y(Xs, y)
        self.n_views_ = len(Xs)
        if type(self.base_estimator) is list:
            if len(self.base_estimator) != self.n_views_:
                raise ValueError(
                    "The length of the estimators should be the same as the"
                    "number of views"
                )
            self.estimators_ = self.base_estimator
        else:
            self.estimators_ = [
                clone(self.base_estimator) for _ in range(self.n_views_)
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

        y : array-like of length (n_samples,), optional (default None)
            Targets for a supervised estimation task

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._prefit(Xs, y)
        for estimator, X in zip(self.estimators_, Xs):
            estimator.fit(X, y)
        return self


class ViewClassifier(BaseWrapper):
    r"""Apply a sklearn classifier to each view of a dataset

    Build a classifier from multiview data by using one
    or more individual scikit-learn classifiers on each view.

    Parameters
    ----------
    base_estimator : a sklearn classifier instance, or a list
        Either a single sklearn classifier that will be applied to each
        view. One clone of the estimator will correspond to each view.
        Otherwise, it should be a list of estimators, of length the number of
        views in the multiview dataset.

    Attributes
    ----------
    n_views_ : int
        The number of views in the input dataset

    estimators_ : list of objects of length n_views_
        The list of classifiers used to predict data labels. If
        self.base_estimator is a single estimator, this is a list containing
        clones of that estimator, otherwise it is one view of
        self.base_estimator.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.compose import ViewClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> Xs, y = load_UCImultifeature()
    >>> clfs = ViewClassifier(LogisticRegression())
    >>> y_hat = clfs.fit(Xs, y).predict(Xs)
    >>> print(y_hat.shape)
    (2000,)
    """
    def predict(self, Xs):
        """
        Return the predicted class labels using majority vote of the
        predictions from each view.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to predict

        Returns
        -------
        y_hat : array-like of shape (n_samples,)
            Predicted class labels for each sample
        """
        check_is_fitted(self)
        Xs, n_views, _, _ = check_Xs(Xs, return_dimensions=True)
        if n_views != self.n_views_:
            raise ValueError(
                f"Multiview input data must have {self.n_views_} views")
        ys = [clf.predict(X) for clf, X in zip(self.estimators_, Xs)]
        return stats.mode(ys, axis=0)[0].squeeze()

    def score(self, Xs, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to predict

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(Xs) w.r.t. y
        """
        return accuracy_score(y, self.predict(Xs), sample_weight=sample_weight)


class ViewTransformer(BaseWrapper, TransformerMixin):
    r"""Apply a sklearn transformer to each view of a dataset

    Build a transformer from multiview dataset to multiview dataset by
    using one or more individual scikit-learn transformers on each view.

    Parameters
    ----------
    base_estimator : a sklearn transformer instance, or a list
        Either a single sklearn transformer that will be applied to each
        view. One clone of the estimator will correspond to each view.
        Otherwise, it should be a list of estimators, of length the number of
        views in the multiview dataset.

    Attributes
    ----------
    n_views_ : int
        The number of views in the input dataset

    estimators_ : list of objects of length n_views_
        The list of transformers used to transform data. If
        self.base_estimator is a single transformer, it is a list containing
        clones of that transformer, otherwise it is a view of
        self.base_estimator.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.compose import ViewTransformer
    >>> from sklearn.decomposition import PCA
    >>> Xs, _ = load_UCImultifeature()
    >>> repeat = ViewTransformer(PCA(n_components=2))
    >>> Xs_transformed = repeat.fit_transform(Xs)
    >>> print(len(Xs_transformed))
    6
    >>> print(Xs_transformed[0].shape)
    (2000, 2)
    """

    def transform(self, Xs, index=None):
        r"""Transform each dataset

        Applies the transform of each transformer on the
        individual views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        index: int or array-like, default=None
            The index or list of indices of the fitted views to which the
            inputted views correspond. If None, there should be as many
            inputted views as the fitted views and in the same order.
            Note that the index parameter is not available in all methods of
            mvlearn yet.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        if index is None:
            index_ = np.arange(len(self.estimators_))
        else:
            index_ = np.copy(index)

        assert len(index_) == len(Xs)

        check_is_fitted(self)
        Xs = check_Xs(Xs)
        Xs_transformed = []
        for estimator, X in zip([self.estimators_[i] for i in index_], Xs):
            Xs_transformed.append(estimator.transform(X))
        return Xs_transformed

    def fit_transform(self, Xs, y=None):
        r"""Fit and transform each dataset

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : numpy.ndarray of shape (n_samples,), optional (default None)
            Target values if a supervised transformation.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        self._prefit(Xs, y)
        Xs_transformed = []
        for estimator, X in zip(self.estimators_, Xs):
            Xs_transformed.append(estimator.fit_transform(X, y))
        return Xs_transformed

    def inverse_transform(self, Xs, index=None):
        r"""Compute the inverse transform of a dataset

        Applies the inverse_transform function of each
        transformer on the individual datasets

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The input data.

        index: int or array-like, default=None
            The index or list of indices of the fitted views to which the
            inputted views correspond. If None, there should be as many
            inputted views as the fitted views and in the same order.
            Note that the index parameter is not available in all methods of
            mvlearn yet.

        Returns
        -------
        Xs_transformed : list of array-likes
            List of length n_views.
            The transformed data.
        """
        check_is_fitted(self)

        if index is None:
            index_ = np.arange(len(self.estimators_))
        else:
            index_ = np.copy(index)

        assert len(index_) == len(Xs)
        Xs = check_Xs(Xs)
        Xs_transformed = []
        for estimator, X in zip([self.estimators_[i] for i in index_], Xs):
            Xs_transformed.append(estimator.inverse_transform(X))
        return Xs_transformed
