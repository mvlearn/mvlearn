"""Singleview function wrapping utilities."""

# Authors: Ronan Perry
#
# License: MIT

from scipy import stats
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from ..utils import check_Xs, check_Xs_y


class ViewClassifier(BaseEstimator):
    r"""Apply a sklearn classifier to each view of a dataset

    Build a transformer from multiview dataset to multiview dataset by
    using individual scikit-learn transformer on each view.

    Parameters
    ----------
    base_transformer : a sklearn transformer instance, or a list
        Either a single sklearn transformer that will be applied to each
        view. One clone of the estimator will correspond to each view.
        Otherwise, it should be a list of estimators, of length the number of
        views in the multiview dataset.

    Attributes
    ----------
    n_views_ : int
        The number of views in the input dataset

    transformers_ : list of objects of length n_views_
        The list of transformer used to transform data. If
        self.base_transformer is a single transformer, it is a list containing
        clones of that transformer, otherwise it is a view of
        self.base_transformer.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.compose import ViewClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> Xs, y = load_UCImultifeature()
    >>> clfs = ViewClassifier(LogisticRegression())
    >>> y_hat = clfs.fit(Xs, y).predict(Xs)
    >>> print(y.shape)
    (2000,)
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def _prefit(self, Xs, y):
        r"""Estimate the attributes of the class.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : array-like of shape (n_samples,)
            Labels for the samples in Xs.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
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

    def fit(self, Xs, y):
        r"""Fit each estimator to the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to.

        y : array-like of shape (n_samples,)
            Labels for the samples in Xs.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._prefit(Xs, y)
        for estimator, X in zip(self.estimators_, Xs):
            estimator.fit(X, y)
        return self

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
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(Xs), sample_weight=sample_weight)
