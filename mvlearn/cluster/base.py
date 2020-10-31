# License: MIT

from abc import abstractmethod
from sklearn.base import BaseEstimator


class BaseCluster(BaseEstimator):
    r'''
    A base class for clustering.
    Parameters
    ----------
    Attributes
    ----------
    See Also
    --------
    '''

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, Xs, y=None):
        '''
        A method to fit clustering parameters to the multiview data.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views to fit the model on.

        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        '''

        return self

    @abstractmethod
    def predict(self, Xs):
        '''
        A method to predict cluster labels of multiview data.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views to cluster.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            Returns the predicted cluster labels for each sample.
        '''

        pass

    def fit_predict(self, Xs, y=None):
        '''
        A method for fitting then predicting cluster assignments.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views to fit the model on.

        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        labels : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.
        '''

        self.fit(Xs)
        labels = self.labels_
        return labels
