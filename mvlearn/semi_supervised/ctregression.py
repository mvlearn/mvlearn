# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implements multi-view co-training regression for 2-view data.


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import random
from ..utils.utils import check_Xs, check_Xs_y_nan_allowed
from .base import BaseCoTrainEstimator


class CTRegressor(BaseCoTrainEstimator):
    r"""
    This class implements the co-training regression for supervised
    and semi supervised learning with the framework as described in [#1CTR]_.
    The best use case is when 2 views of input data are sufficiently
    distinct and independent as detailed in [#1CTR]_. However this can also
    be successfull when a single matrix of input data is given as
    both views and two estimators are choosen
    which are quite different.[#2CTR]_

    In the semi-supervised case, performance can vary greatly, so using
    a separate validation set or cross validation procedure is
    recommended to ensure the regression model has fit well.

    Parameters
    ----------
    estimator1: sklearn object, (only supports KNeighborsRegressor)
        The regressor object which will be trained on view1 of the data.

    estimator2: sklearn object, (only supports KNeighborsRegressor)
        The regressir object which will be trained on view2 of the data.

    k_neighbors: int, optional (default = 5)
        The number of neighbors to be considered for determining the mean
        squared error.

    unlabeled_pool_size: int, optional (default = 50)
        The number of unlabeled_pool samples which will be kept in a
        separate pool for regression and selection by the updated
        regressor at each training iteration.

    num_iter: int, optional (default = 100)
        The maximum number of iteration to be performed

    random_state: int (default = None)
        The seed for fit() method and other class operations

    Attributes
    ----------
    estimator1_ : regressor object, used on view1

    estimator2_ : regressor object, used on view2

    class_name_: string
        The name of the class.

    k_neighbors_ : int
        The number of neighbors to be considered for determining
        the mean squared error.

    unlabeled_pool_size: int
        The number of unlabeled_pool samples which will be kept in a
        separate pool for regression and selection by the updated
        regressor at each training iteration.

    num_iter: int
        The maximum number of iterations to be performed

    n_views : int
        The number of views in the data

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from mvlearn.semi_supervised import CTRegressor
    >>> # X1 and X2 are the 2 views of the data
    >>> X1 = [[0], [1], [2], [3], [4], [5], [6]]
    >>> X2 = [[2], [3], [4], [6], [7], [8], [10]]
    >>> y = [10, 11, 12, 13, 14, 15, 16]
    >>> # Converting some of the labeled values to nan
    >>> y_train = [10, np.nan, 12, np.nan, 14, np.nan, 16]
    >>> knn1 = KNeighborsRegressor(n_neighbors = 2)
    >>> knn2 = KNeighborsRegressor(n_neighbors = 2)
    >>> ctr = CTRegressor(knn1, knn2, k_neighbors = 2, random_state =  42)
    >>> ctr = ctr.fit([X1, X2], y_train)
    >>> pred = ctr.predict([X1, X2])
    >>> print("True value\n{}".format(y))
    True value
    [10, 11, 12, 13, 14, 15, 16]
    >>> print("Predicted value\n{}".format(pred))
    Predicted value
    [10.75 11.25 11.25 13.25 13.25 14.75 15.25]

    Notes
    -----
    Multi-view co-training is most helpful for tasks in semi-supervised
    learning where each view offers unique information not seen in the
    other. As is shown in the example notebooks for using this algorithm,
    multi-view co-training can provide good regression results even
    when number of unlabeled samples far exceeds the number of labeled
    samples. This regressor uses 2 sklearn regressors which work individually
    on each view but which share information and thus result in improved
    performance over looking at the views completely separately.
    The regressor needs to be KNeighborsRegressor,
    as described in the [#1CTR]_.

    Algorithm
    ---------
    Given:

        * a set *L1*, *L2* having labeled training
        samples of each view respectively

        * a set *U* of unlabeled samples

        Create a pool *U'* of examples by choosing examples at random
        from *U*

        * Use *L1* to train a regressor *h1* (``estimator1``) that considers
          only the view1 portion of the data (i.e. Xs[0])
        * Use *L2* to train a regressor *h2* (``estimator2``) that considers
          only the view2 portion of the data (i.e. Xs[1])

        Loop for *T* iterations
            * for each view *j*
                * for each *u* in *U'*
                    * Calculate the neighbors of *u*
                    * Predict the value of *u* using *hj* estimator
                    * create a new estimator *hj'* with same parameters
                    as that of *hj* and train it on the data (*Lj* union *u*)
                    * predict the value of neighbors from estimator *hj*
                    and calculate the mean squared error with respect to
                    original values
                    * predict the value of neighbors from the new
                    estimator *hj'* and calculate the mean squared error
                    with respect to original values
                    * calculate the difference between the two errors
                    * store the error in a list named *deltaj*
            * select the index with maximum positive value from both
            the *delta1* and *delta2*
            * let the indexes selected be *index1* and *index2*
            * Add the *index1* to *L2*
            * Add the *index2* to *L1*
            * Remove the  selected index from *U'* and replenish
            it by taking unlabeled index from *U*
            * Use *L1* to train the regressor *h1*
            * Use *L2* to train the regressor *h2*

    References
    ----------
    [#1CTR] : Semi-Supervised Regression with
            Co-Training by Zhi-Hua Zhou and Ming Li
            https://pdfs.semanticscholar.org/437c/85ad1c05f60574544d31e96bd8e60393fc92.pdf

    [#2CTR] : Goldman, Sally, and Yan Zhou. "Enhancing supervised
            learning with unlabeled data." ICML. 2000.
            http://www.cs.columbia.edu/~dplewis/candidacy/goldman00enhancing.pdf

    """

    def __init__(
        self,
        estimator1=None,
        estimator2=None,
        k_neighbors=5,
        unlabeled_pool_size=50,
        num_iter=100,
        random_state=None
    ):

        # initialize a BaseCTEstimator object
        super().__init__(estimator1, estimator2, random_state)

        # If not given, initialize with default KNeighborsRegrssor
        if estimator1 is None:
            estimator1 = KNeighborsRegressor()
        if estimator2 is None:
            estimator2 = KNeighborsRegressor()

        # Initializing the other attributes
        self.estimator1_ = estimator1
        self.estimator2_ = estimator2
        self.k_neighbors_ = k_neighbors
        self.unlabeled_pool_size = unlabeled_pool_size
        self.num_iter = num_iter

        # Used in fit method while selecting a pool of unlabeled samples
        random.seed(random_state)

        self.n_views = 2
        self.class_name_ = "CTRegressor"

        # checks whether the parameters given is valid
        self._check_params()

    def _check_params(self):
        r"""
        Checks that cotraining parameters are valid. Throws AttributeError
        if estimators are invalid. Throws ValueError if any other parameters
        are not valid. The checks performed are:
            - estimator1 and estimator2 are KNeigborsRegressor
            - k_neighbors_ is positive
            - unlabeled_pool_size is positive
            - num_iter is positive
        """

        # The estimator must be KNeighborsRegressor
        to_be_matched = "KNeighborsRegressor"

        # Taking the str of estimator object
        # returns the class name along with other parameters
        string1 = str(self.estimator1_)
        string2 = str(self.estimator2_)

        # slicing the list to get the name of the estimator
        string1 = string1[: len(to_be_matched)]
        string2 = string2[: len(to_be_matched)]

        if string1 != to_be_matched or string2 != to_be_matched:
            raise AttributeError(
                "Both the estimator needs to be KNeighborsRegressor")

        if self.k_neighbors_ <= 0:
            raise ValueError("k_neighbors must be positive")

        if self.unlabeled_pool_size <= 0:
            raise ValueError("unlabeled_pool_size must be positive")

        if self.num_iter <= 0:
            raise ValueError("number of iterations must be positive")

    def fit(self, Xs, y):
        r"""
        Fit the regressor object to the data in Xs, y.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to train on.

        y : array, shape (n_samples,)
            The target values of the training data. Unlabeled examples
            should have label np.nan.

        Returns
        -------
        self : returns an instance of self
        """

        # check whether Xs contain NaN and both Xs and y
        # are consistent with each other
        Xs, y = check_Xs_y_nan_allowed(
            Xs, y, multiview=True, enforce_views=self.n_views)

        y = np.array(y)

        # Xs contain two view
        X1 = Xs[0]
        X2 = Xs[1]

        # Storing the indexes of the unlabeled samples
        U = [i[0] for i in enumerate(y) if np.isnan(i[1])]

        # Storing the indexes of the labeled samples
        L = [i[0] for i in enumerate(y) if not np.isnan(i[1])]

        # making two true labels for each view
        # So that we can make changes to it without altering original labels
        y1 = y.copy()
        y2 = y.copy()

        # contains the indexes of labeled samples
        L1 = L.copy()
        L2 = L.copy()

        # fitting the estimator object on the train data
        self.estimator1_.fit(X1[L1], y1[L1])
        self.estimator2_.fit(X2[L2], y2[L2])

        # declaring a variable which keeps tracks
        # of the number of iteration performed
        it = 0

        # Randomly selected index of unlabeled data samples
        unlabeled_pool = random.sample(
            U, min(len(U), self.unlabeled_pool_size))

        # Removing the unlabeled samples which were selected earlier
        U = [i for i in U if i not in unlabeled_pool]

        while it < self.num_iter and unlabeled_pool:
            it += 1

            # list of k nearest neighbors for all unlabeled samples
            neighbors1 = self.estimator1_.kneighbors(
                X1[unlabeled_pool],
                n_neighbors=self.k_neighbors_,
                return_distance=False)
            neighbors2 = self.estimator2_.kneighbors(
                X2[unlabeled_pool],
                n_neighbors=self.k_neighbors_,
                return_distance=False)

            # Stores the delta value of each view
            delta1 = []
            delta2 = []

            for i, (u, neigh) in enumerate(zip(unlabeled_pool, neighbors1)):

                # Making a copy of L1 to include the unlabeled index
                new_L1 = L1.copy()
                new_L1.append(u)

                # Predicts the value of unlabeled index
                pred = self.estimator1_.predict(np.expand_dims(X1[u], axis=0))

                # assigning the predicted value to new y
                new_y1 = y1.copy()
                new_y1[u] = pred

                # prediction array before inclusion of unlabeled index
                pred_before_inc = []

                pred_before_inc = self.estimator1_.predict((X1[L1])[neigh])

                # new estimator for training a regressor model on new L1
                new_estimator = KNeighborsRegressor()

                # Setting the same parameters as that of estimator1 object
                new_estimator.set_params(**self.estimator1_.get_params())
                new_estimator.fit(X1[new_L1], new_y1[new_L1])

                # prediction array after inclusion of unlabeled index
                pred_after_inc = []
                pred_after_inc = new_estimator.predict((X1[L1])[neigh])

                mse_before_inc = mean_squared_error(
                    (y1[L1])[neigh], pred_before_inc)

                mse_after_inc = mean_squared_error(
                    (y1[L1])[neigh], pred_after_inc)

                # appending the calculated value to delta1
                delta1.append(mse_before_inc - mse_after_inc)

            for i, (u, neigh) in enumerate(zip(unlabeled_pool, neighbors2)):

                # Making a copy of L2 to include the unlabeled index
                new_L2 = L2.copy()
                new_L2.append(u)

                # Predicts the value of unlabeled index
                pred_before_inc = []

                pred = self.estimator2_.predict(
                    np.expand_dims(X2[u], axis=0))

                # assigning the predicted value to new y
                new_y2 = y2.copy()
                new_y2[u] = pred

                # prediction array before inclusion of unlabeled index
                pred_before_inc = self.estimator2_.predict((X2[L2])[neigh])

                # new estimator for training a regressor model on new L2
                new_estimator = KNeighborsRegressor()

                # Setting the same parameters as that of estimator2 object
                new_estimator.set_params(**self.estimator2_.get_params())
                new_estimator.fit(X2[new_L2], new_y2[new_L2])

                # prediction array after inclusion of unlabeled index
                pred_after_inc = []
                pred_after_inc = new_estimator.predict((X2[L2])[neigh])

                mse_before_inc = mean_squared_error(
                    (y2[L2])[neigh], pred_before_inc)

                mse_after_inc = mean_squared_error(
                    (y2[L2])[neigh], pred_after_inc)

                # appending the calculated value to delta2
                delta2.append(mse_before_inc - mse_after_inc)

            delta1_index = np.argsort(delta1)
            delta2_index = np.argsort(delta2)

            # list containing the indexes to be included
            to_include1 = []
            to_include2 = []

            """
            If the length of both the delta's is equal to 1 then
            include the corresponding index whose value is positive and
            greater than the other values.
            Else selecting the indexes which have postive and maximum
            value from each delta's and incase both the indexes are equal
            then look at the second best positive value.
            The indexes which are selected from delta1
            will be added to the labels of the estimator2 object.
            Similarly, the indexes which are selected from delta2
            will be added to the labels of the estimator1 object.
            """
            if delta1_index.shape[0] == 1 and delta2_index.shape[0] == 1:

                if delta1[0] > 0 and delta2[0] > 0:
                    if delta1[0] >= delta2[0]:
                        L2.append(unlabeled_pool[0])
                        to_include2.append(0)
                    else:
                        L1.append(unlabeled_pool[0])
                        to_include1.append(0)

                elif delta1[0] > 0:
                    L2.append(unlabeled_pool[0])
                    to_include2.append(0)

                elif delta2[0] > 0:
                    L1.append(unlabeled_pool[0])
                    to_include1.append(0)

            else:

                # Top two indexes from each delta
                index1_1, index1_2 = delta1_index[-1], delta1_index[-2]
                index2_1, index2_2 = delta2_index[-1], delta2_index[-2]

                if index1_1 != index2_1:
                    if delta1[index1_1] > 0:
                        L2.append(unlabeled_pool[index1_1])
                        to_include2.append(index1_1)

                    if delta2[index2_1] > 0:
                        L1.append(unlabeled_pool[index2_1])
                        to_include1.append(index2_1)

                else:
                    if delta1[index1_1] > 0 and delta2[index2_1] > 0:
                        if delta1[index1_1] >= delta2[index2_1]:
                            L2.append(unlabeled_pool[index1_1])
                            to_include2.append(index1_1)
                            if delta2[index2_2] > 0:
                                L1.append(unlabeled_pool[index2_2])
                                to_include1.append(index2_2)

                        else:
                            L1.append(unlabeled_pool[index2_1])
                            to_include1.append(index2_1)
                            if delta1[index1_2] > 0:
                                L2.append(unlabeled_pool[index1_2])
                                to_include2.append(index1_2)

                    elif delta1[index1_1] > 0:
                        L2.append(unlabeled_pool[index1_1])
                        to_include2.append(index1_1)

                    elif delta2[index2_1] > 0:
                        L1.append(unlabeled_pool[index2_1])
                        to_include1.append(index2_1)

            # break if to_include1 and to_include2 are empty
            if len(to_include1) == 0 and len(to_include2) == 0:
                break

            # including the selected index
            for i in to_include1:
                pred = self.estimator2_.predict(
                    np.expand_dims(X2[unlabeled_pool[i]], axis=0))
                y1[unlabeled_pool[i]] = pred

            # including the selected index
            for i in to_include2:
                pred = self.estimator1_.predict(
                    np.expand_dims(X1[unlabeled_pool[i]], axis=0))
                y2[unlabeled_pool[i]] = pred

            # Currently to_include contains the index of unlabeled samples
            # in the order in which they are stored in unlabeled_pool
            # Converting them to the value which unlabeled_pool stores
            # example unlabeled_pool = [10, 15, 17]
            # current to_include = [1, 2]
            # updated to_include = [15, 17]
            to_include1 = [unlabeled_pool[i] for i in to_include1]
            to_include2 = [unlabeled_pool[i] for i in to_include2]

            # removing the selected index
            unlabeled_pool = [
                u for u in unlabeled_pool
                if (u not in to_include1) and (u not in to_include2)]

            # replenishing the unlabeled pool
            for u in U:
                if len(unlabeled_pool) < self.unlabeled_pool_size:
                    if u not in unlabeled_pool:
                        unlabeled_pool.append(u)
                else:
                    break

            U = [i for i in U if i not in unlabeled_pool]

            # fitting the model on new train data
            self.estimator1_.fit(X1[L1], y1[L1])
            self.estimator2_.fit(X2[L2], y2[L2])

        return self

    def predict(self, Xs):
        r"""
        Predict the values of the samples in the two input views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to predict.

        Returns
        -------
        y_pred : array-like (n_samples,)
            The average of the predictions from both estimators is returned
        """
        Xs = check_Xs(Xs, multiview=True, enforce_views=self.n_views)

        X1 = Xs[0]
        X2 = Xs[1]

        # predicting the value of each view
        pred1 = self.estimator1_.predict(X1)
        pred2 = self.estimator2_.predict(X2)

        # Taking the average of the predicted value and returning it
        return (pred1 + pred2) * 0.5
