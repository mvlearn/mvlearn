# python version 3.7.6

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import random
from scipy.spatial.distance import minkowski
from ..utils.utils import check_Xs, check_Xs_y_nan_allowed
from .base import BaseCoTrainEstimator

# Implements Multi view Co training Regression for 2 view data

class CTRegressor(BaseCoTrainEstimator):
    r"""
    This class implements the co-training regression for supervised
    and semisupervised learning with the framework as described in [#1CTR].
    The best use case is when nthe 2 views of input data are sufficiently
    distinct and independent as detailed in [#1CTR]. However this can also
    be successfull when na single matrix of input data is given as 
    both views and two estimators are choosen which are quite different.[#2CTR]

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

    supsample_size: int, optional (default = 50)
        The size of unlabelled subsample

    n_iter: int, optional (default = 100)
        The maximum number of iteration to be performed
    
    selected_size: int, optional (default = 5)
        The maximum number of samples to be selected from the pool of unlabelled samples
    
    random_state: int (defalut = None)
        The seed for fit() method and other class operations
    
    Attributs
    ---------
    estimator1 : regressor object, used on view1

    estimator2 : regressor object, used on view2

    neighbors_size : int
        The number of neighbours to be considered for determining the mean squared error.

    supsample_size: int
        The size of unlabelled subsample

    n_iter: int
        The maximum number of iteration to be performed
    
    selected_size: int
        The maximum number of samples to be selected from the pool of unlabelled samples
    
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
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    >>> # Dowload link of data https://www.kaggle.com/shivachandel/kc-house-data
    >>> data = pd.read_csv("D:/Downloads/housesalesprediction/kc_house_data.csv")   # input your own data path
    >>> data = data.drop(columns = ['id', 'date'])
    >>> X = data.drop(columns = ['price'])
    >>> Y = data['price']
    >>> X1 = X.iloc[:, : int(X.shape[1]/2)]
    >>> X2 = X.iloc[:, int(X.shape[1]/2) :]
    >>> X1_train, X1_test, y_train, y_test = train_test_split(X1, Y, random_state = 42, test_size = 0.30)
    >>> X2_train, X2_test, y_train, y_test = train_test_split(X1, Y, random_state = 42, test_size = 0.30)
    >>> # Supervised learning with a single view of data and 2 estimator types
    >>> estimator1 = KNeighborsRegressor(p = 3)
    >>> estimator2 = KNeighborsRegressor(p = 5)
    >>> ctr = CTRegressor(estimator1, estimator2, random_state=1)
    >>> # Use the same matrix for each view
    >>> ctr = ctr.fit([X1_train, X1_train], y_train)
    >>> preds = ctr.predict([X1_test, X1_test])
    >>> print("RMSE loss: {}".format(np.sqrt(mean_squared_error(y_test, preds))))
    RMSE loss: 281801.0172792788

    Notes
    -----
    Multi-view co-training is most helpful for tasks in semi-supervised
    learning where each view offers unique information not seen in the
    other. As is shown in the example notebooks for using this algorithm,
    multi-view co-training can provide good regression results even
    when number of unlabeled samples far exceeds the number of labeled
    samples. This regressor uses 2 slearn regressors which work individually
    on each view but which share information and thus result in improved
    performance over looking at the views completely separately. 
    The regressor needs to be KNearestRegressor, as described in the [#1CTR].

    Algorithm
    ---------
    Given:

        * a set *L1*, *L2* having labeled training samples of each view respectively

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
                    * Calculate the *k* nearest neighbour of *u* (lets call the list of neighbours as *omega*)
                    * Train a new regressor model *hj'* with same parameters as *hj* on the *Lj* union *u* with label as *y_hat*
                    * for i in *omega*
                        * Calculate the predicted value of i and take the differene between initial predicted value from *hj* 
                        and final predicted value from  new *hj'* for each i
                        * Add the calculated value to a list named as *delta*
                * sort the list in descending order of values and select those indexes whose difference turned out to be positive
                * let the list of indexes selected be *pi_j*
                * remove the selected indexes from *U'* and replenish it
            * Add the selected index *pi_1* to the other regressor training example i.e. *L2*
            * Add the selected index *pi_2* to the other regressor training example i.e. *L1*
            * Use *L1* to train the regressor *h1*
            * Use *L2* to train the regressor *h2*
        
    Reference
    ---------
    [#1CTR] : Semi-Supervised Regression with Co-Training by Zhi-Hua Zhou and Ming Li
            https://pdfs.semanticscholar.org/437c/85ad1c05f60574544d31e96bd8e60393fc92.pdf

    [#2CTC] : Goldman, Sally, and Yan Zhou. "Enhancing supervised
            learning with unlabeled data." ICML. 2000.
            http://www.cs.columbia.edu/~dplewis/candidacy/goldman00enhancing.pdf

    """
    def __init__(
        self,
        estimator1 = None,
        estimator2 = None,
        neighbors_size = 5,
        subsample_size = 50,
        n_iter = 100,
        selected_size = 5,
        random_state = None
        ):

        # initialize a BaseCTEstimator object
        super().__init__(estimator1, estimator2, random_state)

        # If not given initialize with default KNeighborsRegrssor
        if estimator1 is None:
            estimator1 = KNeighborsRegressor()
        if estimator2 is None:
            estimator2 = KNeighborsRegressor()

        # Initializing the other attributes
        self.estimator1 = estimator1
        self.estimator2 = estimator2
        self.neighbors_size = neighbors_size
        self.subsample_size = subsample_size
        self.n_iter = n_iter
        self.selected_size = selected_size
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
            - neighbors_size is positive
            - subsample_size is positive
            - n_iter is positive
            - selected_size is positive
        """

        # The estimator must be KNeighborsRegressor
        to_be_matched = "KNeighborsRegressor"

        #Taking the str of a class returns the class name along with other parameters
        string1 = str(self.estimator1)

        #slicing the list to get the name of the estimator
        string1 = string1[:len(to_be_matched)]

        #Taking the str of a class returns the class name along with other parameters   
        string2 = str(self.estimator2)

        #slicing the list to get the name of the estimator
        string2 = string2[:len(to_be_matched)]
        
        if string1 != to_be_matched or string2 != to_be_matched:
            raise AttributeError("Both the estimator need to be KNeighborsRegressor")

        if self.neighbors_size <= 0:
            raise ValueError("neighbors size must be positive")

        if self.subsample_size <= 0:
            raise ValueError("subsample size must be positive")

        if self.n_iter <= 0:
            raise ValueError("number of iterations must be positive")

        if self.selected_size <= 0:
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

        # check whether Xs contain NaN and both Xs and y are consistent woth each other
        check_Xs_y_nan_allowed(Xs, y,  multiview = True, enforce_views = self.n_views)

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
        # So that we can make changes to it without altering the original labels
        y1 = y
        y2 = y

        # contains the indexes of labelled sample
        not_null1 = L
        not_null2 = L

        # fitting the estimator object on the train data 
        self.estimator1.fit(X1.iloc[not_null1], y1.iloc[not_null1])
        self.estimator2.fit(X2.iloc[not_null2], y2.iloc[not_null2])
        
        # declaring a variable which keeps tracks of the number of iteration performed
        it = 0

        while it < self.n_iter and U:
            # print("Number of iteration done {} \nSize of unlabelled sample {}".format(it, len(U)))
            it += 1
            random.seed(self.random_state)

            # Taking a subsample from unlabelled indexes
            u = random.sample(U, min(len(U), self.subsample_size))

            # constains a list of [delta, index, predicted_value]
            # and after inclusion of unlabelled index in the training data
            # value for each subsampled index
            delta1 = []

            # contains the index of unlabelled sample which will be 
            # included in the other regressor object for further training
            to_include1 = []

            for i in u:
                
                # delta is defines as the differece between the predicted value of k nearest neighbour before 
                delta = 0

                # list of k nearest neighbour for the unlabelled sample
                omega = self.find_k_nearest_neighbour(X1, y1, i)

                for j in omega:

                    # prediction value of each neighbour before including the unlabelled sample in the training view
                    before_pred = self.estimator1.predict(np.expand_dims(X1.iloc[j], axis = 0))

                    # predicted value of unlabelled sample
                    pred = self.estimator1.predict(np.expand_dims(X1.iloc[i], axis = 0))

                    # include the predicted value of unlabelled sample
                    y1.iloc[i] = pred

                    # a temporary estimator which fits and predicts the value of 
                    # each neighbours after the inclusion of unlabelled sample
                    temp_estimator = self.estimator1
                    
                    # including the unlablled sample index for training
                    not_null1.append(i)

                    temp_estimator.fit(X1.iloc[not_null1], y1.iloc[not_null1])
                    after_pred = temp_estimator.predict(np.expand_dims(X1.iloc[j], axis = 0))

                    # calculating the value of delta
                    delta += pow(y1.iloc[j] - before_pred, 2) - pow(y1.iloc[j] - after_pred, 2)

                    # making the unlabelled sample again unlabelled and removing it from not_null list
                    y1.iloc[i] = np.nan
                    not_null1.pop()

                # appending the index, delta, pred value in list
                delta1.append([i, delta, pred])

            # sorting the delta1 with descending order of delta value
            delta1 = sorted(delta1, key = lambda x : -x[1])

            # counts the number of delta value stored
            count = 0

            # counts the index of delta1
            index = 0

            while index < len(delta1) and count < self.selected_size and delta1:

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
                if len(u) < self.subsample_size:
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
                # delta is defines as the differece between the predicted value of k nearest neighbour before 
                delta = 0

                # list of k nearest neighbour
                omega = self.find_k_nearest_neighbour(X2, y2, i)

                for j in omega:
                    # prediction value of each neighbour before including the unlabelled sample in the training view
                    before_pred = self.estimator2.predict(np.expand_dims(X2.iloc[j], axis = 0))

                    # predicted value of unlabelled sample
                    pred = self.estimator2.predict(np.expand_dims(X2.iloc[i], axis = 0))

                    # include the predicted value of unlabelled sample
                    y2.iloc[i] = pred

                    # a temporary estimator which fits and predicts the value of 
                    # each neighbours after the inclusion of unlabelled sample
                    temp_estimator = self.estimator2

                    # including the unlablled sample index for training
                    not_null2.append(i)

                    temp_estimator.fit(X2.iloc[not_null2], y2.iloc[not_null2])
                    after_pred = temp_estimator.predict(np.expand_dims(X2.iloc[j], axis = 0))

                    # calculating the delta value
                    delta += pow(y2.iloc[j] - before_pred, 2) - pow(y2.iloc[j] - after_pred, 2)

                    # making the unlabelled sample again unlabelled and removing it from not_null list
                    y2.iloc[i] = np.nan
                    not_null2.pop()

                # appending the index, delta, pred value in list
                delta2.append([i, delta, pred])

            # sorting the delta1 with descending order of delta value
            delta2 = sorted(delta2, key = lambda x : -x[1])

            # counts the number of delta value stored
            count = 0

            # counts the index of delta2
            index = 0

            while index < len(delta2) and count < self.selected_size and delta2:
                # if delta2's delta value is not positive break the loop
                if delta2[index][1] <= 0:
                    break
                else:
                    count += 1
                    to_include2.append(delta2[index][0])
                index += 1

            # removing the unlabelled samples 
            U = [i for i in U if (i not in to_include2) and (i not in to_include1)]
            # print("\nto_include1 size {}\tto_include2 size {}".format(len(to_include1), len(to_include2)))

            # if both lists are empty break the iterations
            if (not to_include1) and (not to_include2):
                break
            
            # include the unlablled sample into the other regressor object training sample
            not_null1.extend(to_include2)
            not_null2.extend(to_include1)

            # label the predicted value of each unlabelled sample 
            for i, num in enumerate(to_include1):
                y2.iloc[num] = delta1[i][2]

            for i, num in enumerate(to_include2):
                y1.iloc[num] = delta2[i][2]

            # fit the estimator again on the new training set
            self.estimator1.fit(X1.iloc[not_null1], y1.iloc[not_null1])
            self.estimator2.fit(X2.iloc[not_null2], y2.iloc[not_null2])

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
        pred1 = self.estimator1.predict(X1)
        pred2 = self.estimator2.predict(X2)

        # Taking the average of the predicted value and returning it
        return ((pred1 + pred2) * 0.5)
    
    def find_k_nearest_neighbour(self, X, y, index):
        r"""
        Finds the K nearest neighbour of the unlabelled point.

        Parameters
        ----------
        X : pandas Dataframe, reprsents a particular view
        y : pandas DataFrame, represent true value
        index : int, represents the index of unlabelled sample whose neighbours needs to be calculated

        Returns
        -------
        omega : list, returs a list of K nearest neighbour
        """

        # List of neighbours , contains a list of index and its corresponding distance from the unlabelled sample
        dist = []

        length = X.shape[0]

        for i in range(0, length):
            if np.isnan(y.iloc[i]):
                continue
            distance = minkowski(np.array(X.iloc[i]), np.array(X.iloc[index]), 2)
            dist.append([i, distance])

        #sorting the dist in ascending order of distance from unlabelled sample
        dist = sorted(dist)

        # storing the index of neigbours with increasing distance from unlabelled sample
        omega = [index_[0] for index_ in dist]

        #returns the list of K nearest neighbour
        return omega[:min(self.neighbors_size, len(omega))]