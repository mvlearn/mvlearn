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


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import random
from scipy.spatial.distance import minkowski
from ..utils.utils import check_Xs, check_Xs_y_nan_allowed
from .base import BaseCoTrainEstimator


class CTRegressor(BaseCoTrainEstimator):
    r"""
    This class implements the co-training regression for supervised
    and semisupervised learning with the framework as described in [#1CTR]_.
    The best use case is when nthe 2 views of input data are sufficiently
    distinct and independent as detailed in [#1CTR]_. However this can also
    be successfull when na single matrix of input data is given as
    both views and two estimators are choosen
    which are quite different.[#2CTR]_

    In the semi-supervised case, performance can vary greatly, so using
    a separate validation set or cross validation procedure is
    recommended to ensure the regression model has fit well.

    Parameters
    ----------
    estimator1: sklearn object, (currently only supports K Nearest Neighbour)
        The regressor object which will be trained on view1 of the data.

    estimator2: sklearn object, (currently only support K Nearest Neighbour)
        The regressir object which will be trained on view2 of the data.

    neighbors_size: int, optional (default = 5)
        The number of neighbours to be considered for determining the mean
        squared error.

    unlabelled_pool_size: int, optional (default = 50)
        The size of unlabelled subsample

    num_iter: int, optional (default = 100)
        The maximum number of iteration to be performed

    unlabelled_subsample_pool_size: int, optional (default = 5)
        The maximum number of samples to be selected from the
        pool of unlabelled samples

    random_state: int (default = None)
        The seed for fit() method and other class operations

    Attributs
    ---------
    estimator1_ : regressor object, used on view1

    estimator2_ : regressor object, used on view2

    neighbors_size_ : int
        The number of neighbours to be considered for determining
        the mean squared error.

    unlabelled_pool_size: int
        The size of unlabelled subsample

    num_iter: int
        The maximum number of iteration to be performed

    unlabelled_subsample_pool_size: int
        The maximum number of samples to be selected from
        the pool of unlabelled samples

    random_state: int
        The seed for fit() method and other class operations

    n_views : int
        The number of views in the data

    Examples
    --------
    >>> # Supervised learning of single-view data with 2 distinct estimators
    >>> from mvlearn.semi_supervised import CTRegressor
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from sklearn.model_selection import train_test_split as split_
    >>> from sklearn.metrics import mean_squared_error
    >>> # Download link of data
    >>> # https://www.kaggle.com/shivachandel/kc-house-data
    >>> path = "D:/Downloads/housesalesprediction/kc_house_data.csv"
    >>> data = pd.read_csv(path)
    >>> data = data.drop(columns = ['id', 'date'])
    >>> X = data.drop(columns = ['price'])
    >>> Y = data['price']
    >>> X1 = X[:, : int(X.shape[1]/2)]
    >>> X2 = X[:, int(X.shape[1]/2) :]
    >>> X1_train, X1_test, y_train, y_test = split_(X1, Y, random_state = 42)
    >>> X2_train, X2_test, y_train, y_test = split_(X1, Y, random_state = 42)
    >>> # Supervised learning with a single view of data and 2 estimator types
    >>> estimator1 = KNeighborsRegressor(p = 3)
    >>> estimator2 = KNeighborsRegressor(p = 5)
    >>> ctr = CTRegressor(estimator1, estimator2, random_state=1)
    >>> # Use the same matrix for each view
    >>> ctr = ctr.fit([X1_train, X1_train], y_train)
    >>> preds = ctr.predict([X1_test, X1_test])
    >>> error = np.sqrt(mean_squared_error(y_test, preds))
    >>> print("RMSE loss: {}".format(error))
    RMSE loss: 282649.4023784183

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
    The regressor needs to be KNeigborsRegressor, as described in the [#1CTR]_.

    Algorithm
    ---------
    Given:

        * a set *L1*, *L2* having labeled training
        samples of each view respectively

        * a set *U* of unlabeled samples (with 2 views)

    Create a pool *U'* of examples at random from *U*

        * Use *L1* to train a regressor *h1* (``estimator1``) that considers
          only the view 1 portion of the data (i.e. Xs[0])
        * Use *L2* to train a regressor *h2* (``estimator2``) that considers
          only the view 2 portion of the data (i.e. Xs[1])

        Loop for *T* iterations
            * for each viewj
                * for each *u* in *U'*
                    * Predict the value *y_hat* of *u* by *hj*
                    * Calculate the *k* nearest neighbour of *u*
                    (lets call the list of neighbours as *omega*)
                    * Train a new regressor model *hj'* with same parameters
                    as *hj* on the *Lj* union *u* with label as *y_hat*
                    * for i in *omega*
                        * Calculate the predicted value of i and take the
                        difference between initial predicted value from *hj*
                        and final predicted value from  new *hj'* for each i
                        * Add the calculated value to a list named as *delta*
                * sort the list in descending order of values and select
                those indexes whose difference turned out to be positive
                * let the list of indexes selected be *pi_j*
                * remove the selected indexes from *U'* and replenish it
            * Add the selected index *pi_1* to the other
            regressor training example i.e. *L2*
            * Add the selected index *pi_2* to the other
            regressor training example i.e. *L1*
            * Use *L1* to train the regressor *h1*
            * Use *L2* to train the regressor *h2*
    Reference
    ---------
    [#1CTR]_ : Semi-Supervised Regression with
            Co-Training by Zhi-Hua Zhou and Ming Li
            https://pdfs.semanticscholar.org/437c/85ad1c05f60574544d31e96bd8e60393fc92.pdf

    [#2CTR]_ : Goldman, Sally, and Yan Zhou. "Enhancing supervised
            learning with unlabeled data." ICML. 2000.
            http://www.cs.columbia.edu/~dplewis/candidacy/goldman00enhancing.pdf

    """

    def __init__(
        self,
        estimator1=None,
        estimator2=None,
        neighbors_size=5,
        unlabelled_pool_size=50,
        num_iter=100,
        selected_size=5,
        random_state=None
    ):

        # initialize a BaseCTEstimator object
        super().__init__(estimator1, estimator2, random_state)

        # If not given initialize with default KNeighborsRegrssor
        if estimator1 is None:
            estimator1 = KNeighborsRegressor()
        if estimator2 is None:
            estimator2 = KNeighborsRegressor()

        # Initializing the other attributes
        self.estimator1_ = estimator1
        self.estimator2_ = estimator2
        self.neighbors_size_ = neighbors_size
        self.unlabelled_pool_size = unlabelled_pool_size
        self.num_iter = num_iter
        self.unlabelled_subsample_pool_size = selected_size
        self.random_state = random_state
        self.n_views = 2

        # checks whether the parameters given is valid
        self._check_params()

    def _check_params(self):
        r"""
        Checks that cotraining parameters are valid. Throws AttributeError
        if estimators are invalid. Throws ValueError if any other parameters
        are not valid. The checks performed are:
            - estimator1 and estimator2 are KNeigborsRegressor
            - neighbors_size_ is positive
            - unlabelled_pool_size is positive
            - num_iter is positive
            - unlabelled_subsample_pool_size is positive
        """

        # The estimator must be KNeighborsRegressor
        to_be_matched = "KNeighborsRegressor"

        # Taking the str of a class returns the class name
        # along with other parameters
        string1 = str(self.estimator1_)

        # slicing the list to get the name of the estimator
        string1 = string1[: len(to_be_matched)]

        # Taking the str of a class returns the class name
        # along with other parameters
        string2 = str(self.estimator2_)

        # slicing the list to get the name of the estimator
        string2 = string2[: len(to_be_matched)]

        if string1 != to_be_matched or string2 != to_be_matched:
            raise AttributeError(
                "Both the estimator need to be KNeighborsRegressor")

        if self.neighbors_size_ <= 0:
            raise ValueError("neighbors size must be positive")

        if self.unlabelled_pool_size <= 0:
            raise ValueError("subsample size must be positive")

        if self.num_iter <= 0:
            raise ValueError("number of iterations must be positive")

        if self.unlabelled_subsample_pool_size <= 0:
            raise ValueError("selected size must be positive")

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
            The labels of the training data. Unlabeled_pool examples should
            have label np.nan.

        Returns
        -------
        self : returns an instance of self
        """

        # check whether Xs contain NaN and both Xs and y
        # are consistent woth each other
        check_Xs_y_nan_allowed(
            Xs, y, multiview=True, enforce_views=self.n_views)

        distance1, distance2 = self._calc_distance(Xs)

        # Xs contain two views
        view1 = Xs[0]
        view2 = Xs[1]

        # Storing the indexes of the unlabelled samples
        U = [i[0] for i in enumerate(y) if np.isnan(i[1])]

        # Storing the indexes of the labelled samples
        L = [i[0] for i in enumerate(y) if not np.isnan(i[1])]

        X1 = view1
        X2 = view2

        # making two true labels for each view
        # So that we can make changes to it without altering original labels
        y1 = y
        y2 = y

        # contains the indexes of labelled sample
        not_null1 = L
        not_null2 = L

        # fitting the estimator object on the train data
        self.estimator1_.fit(X1[not_null1], y1[not_null1])
        self.estimator2_.fit(X2[not_null2], y2[not_null2])

        # declaring a variable which keeps tracks
        # of the number of iteration performed
        it = 0

        while it < self.num_iter and U:
            # print("Number of iteration done {} \
            # \nSize of unlabelled sample {}".format(it, len(U)))
            it += 1
            random.seed(self.random_state)

            # Taking a subsample from unlabelled indexes
            u = random.sample(U, min(len(U), self.unlabelled_pool_size))

            # constains a list of [delta, index, predicted_value]
            # and after inclusion of unlabelled index in the training data
            # value for each subsampled index
            delta1 = []

            # contains the index of unlabelled sample which will be
            # included in the other regressor object for further training
            to_include1 = []

            for i in u:

                # delta is defines as the differece between the predicted
                # value of k nearest neighbour before
                delta = 0

                # list of k nearest neighbour for the unlabelled sample
                omega = self._find_k_nearest_neighbor(y1, i, distance1)

                for j in omega:

                    # prediction value of each neighbour before including
                    # the unlabelled sample in the training view
                    before_pred = self.estimator1_.predict(
                        np.expand_dims(X1[j], axis=0)
                    )

                    # predicted value of unlabelled sample
                    pred = self.estimator1_.predict(
                        np.expand_dims(X1[i], axis=0))

                    # include the predicted value of unlabelled sample
                    y1[i] = pred

                    # a temporary estimator which fits and predicts the value
                    # of each neighbours after inclusion of unlabelled sample
                    temp_estimator_ = self.estimator1_

                    # including the unlablled sample index for training
                    not_null1.append(i)

                    temp_estimator_.fit(X1[not_null1], y1[not_null1])
                    after_pred = temp_estimator_.predict(
                        np.expand_dims(X1[j], axis=0)
                    )

                    # calculating the value of delta
                    delta += pow(y1[j] - before_pred, 2) - pow(
                        y1[j] - after_pred, 2
                    )

                    # making the unlabelled sample again unlabelled
                    # and removing it from not_null list
                    y1[i] = np.nan
                    not_null1.pop()

                # appending the index, delta, pred value in list
                delta1.append([i, delta, pred])

            # sorting the delta1 with descending order of delta value
            delta1 = sorted(delta1, key=lambda x: -x[1])

            # counts the number of delta value stored
            count = 0

            # counts the index of delta1
            index = 0

            while index < len(delta1) and\
                    count < self.unlabelled_subsample_pool_size and\
                    delta1:

                # if delta1's delta value is not positive break the loop
                if delta1[index][1] <= 0:
                    break
                else:
                    count += 1
                    to_include1.append(delta1[index][0])
                index += 1

            # removing the unlabelled sample which were selected
            u = [i for i in u if i not in to_include1]

            # Replenishing the subsample of unlabelled data
            for m in U:
                if len(u) < self.unlabelled_pool_size:
                    if m not in u:
                        u.append(m)
                else:
                    break

            # constains a list of [delta, index, predicted_value]
            # and after inclusion of unlabelled index in the training data
            # value for each subsampled index
            delta2 = []
            to_include2 = []

            for i in u:
                # delta is defines as the differece between the
                # predicted value of k nearest neighbour before
                delta = 0

                # list of k nearest neighbour
                omega = self._find_k_nearest_neighbor(y2, i, distance2)

                for j in omega:
                    # prediction value of each neighbour before
                    # including the unlabelled sample in the training view
                    before_pred = self.estimator2_.predict(
                        np.expand_dims(X2[j], axis=0)
                    )

                    # predicted value of unlabelled sample
                    pred = self.estimator2_.\
                        predict(np.expand_dims(X2[i], axis=0))

                    # include the predicted value of unlabelled sample
                    y2[i] = pred

                    # a temporary estimator which fits and
                    # predicts the value of each neighbours
                    # after the inclusion of unlabelled sample
                    temp_estimator_ = self.estimator2_

                    # including the unlablled sample index for training
                    not_null2.append(i)

                    temp_estimator_.fit(X2[not_null2], y2[not_null2])
                    after_pred = temp_estimator_.predict(
                        np.expand_dims(X2[j], axis=0)
                    )

                    # calculating the delta value
                    delta += pow(y2[j] - before_pred, 2) - pow(
                        y2[j] - after_pred, 2
                    )

                    # making the unlabelled sample again unlabelled
                    # and removing it from not_null list
                    y2[i] = np.nan
                    not_null2.pop()

                # appending the index, delta, pred value in list
                delta2.append([i, delta, pred])

            # sorting the delta1 with descending order of delta value
            delta2 = sorted(delta2, key=lambda x: -x[1])

            # counts the number of delta value stored
            count = 0

            # counts the index of delta2
            index = 0

            while index < len(delta2) and\
                    count < self.unlabelled_subsample_pool_size and\
                    delta2:
                # if delta2's delta value is not positive break the loop
                if delta2[index][1] <= 0:
                    break
                else:
                    count += 1
                    to_include2.append(delta2[index][0])
                index += 1

            # removing the unlabelled samples
            U = [
                i for i in U
                if (i not in to_include2)
                and
                (i not in to_include1)
                ]
            # print("\nto_include1 size {}\tto_include2 size {}"
            #   .format(len(to_include1), len(to_include2)))

            # if both lists are empty break the iterations
            if (not to_include1) and (not to_include2):
                break

            # include the unlablled sample into the
            # other regressor object training sample
            not_null1.extend(to_include2)
            not_null2.extend(to_include1)

            # label the predicted value of each unlabelled sample
            for i, num in enumerate(to_include1):
                y2[num] = delta1[i][2]

            for i, num in enumerate(to_include2):
                y1[num] = delta2[i][2]

            # fit the estimator again on the new training set
            self.estimator1_.fit(X1[not_null1], y1[not_null1])
            self.estimator2_.fit(X2[not_null2], y2[not_null2])

        return self

    def predict(self, Xs):
        r"""
        Predict the classes of the examples in the two input views.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to predict.

        Returns
        -------
        y_pred : array-like (n_samples,)
            The predicted value of each input example. The average of
            the two predicited values is returned
        """

        X1 = Xs[0]
        X2 = Xs[1]

        # predicting the value of each view
        pred1 = self.estimator1_.predict(X1)
        pred2 = self.estimator2_.predict(X2)

        # Taking the average of the predicted value and returning it
        return (pred1 + pred2) * 0.5

    def _find_k_nearest_neighbor(self, y, index, distance):
        r"""
        Finds the K nearest neighbor of the unlabelled point.

        Parameters
        ----------
        y : represent true value of a view
        index : int, represents the index of unlabelled sample
            whose neighbours needs to be calculated
        distance : dictionary with values as a list of index and distance
            represents the distance of one index with other indexes

        Returns
        -------
        omega : list, returs a list of K nearest neighbor
        """

        omega = []

        # Calculates the distance of one index with every other index
        for i, val in enumerate(y):
            if np.isnan(val) or i == index:
                continue
            else:
                omega.append(distance[index][i][0])

        return omega[:min(len(omega, self.neighbors_size_))]
    
    def _calc_distance(self, Xs):
        r"""
        Calculates the distance of every in a view with every other index.
        The distances are then stored in a dictionary, where every key has list of indexes
        and distance as value.

        Parameters
        ----------
        Xs : list, contains each view of training sample

        Returns
        -------
        distance : tuple, contains a dictionay of respective distances from each view
        """

        X1 = Xs[0]
        X2 = Xs[1]

        # Dictionary having a list of index and distance as value for each key
        distance1 = {}

        for i, arr_i in enumerate(X1):
            for j, arr_j in enumerate(X1):

                # Calculates the  minkowski distance between two array
                dist = minkowski(arr_i, arr_j, 2)
                distance1[i].append([j, dist])

            # sorting the value of each key according to its distance with other indexs
            distance1[i] = sorted(distance1[i])

        # Dictionary having a list of index and distance as value for each key
        distance2 = {}

        for i, arr_i in enumerate(X2):
            for j, arr_j in enumerate(X2):

                # Calculates the  minkowski distance between two array
                dist = minkowski(arr_i, arr_j, 2)
                distance2[i].append([j, dist])

            # sorting the value of each key according to its distance with other indexs
            distance2[i] = sorted(distance2[i])

        # Tuple for storing both the dictionaries
        distance = (distance1, distance2)

        return distance
