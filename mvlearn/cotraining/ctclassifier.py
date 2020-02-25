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
# Implements multi-view co-training classification for 2-view data.

from .base import BaseCoTrainEstimator
import numpy as np
from sklearn.naive_bayes import GaussianNB
from ..utils.utils import check_Xs, check_Xs_y_nan_allowed


class CTClassifier(BaseCoTrainEstimator):
    r"""
    Co-Training Classifier

    This class implements the co-training classifier for semi-supervised
    learning with the framework as described in [#1CTC]_. This should ideally
    be used on 2 views of the input data which satisfy the 3 conditions for
    multi-view co-training (sufficiency, compatibility, conditional
    independence) as detailed in [#1CTC]_. Extends BaseCoTrainEstimator.

    Parameters
    ----------
    estimator1 : classifier object, (default=sklearn GaussianNB)
        The classifier object which will be trained on view 1 of the data.
        This classifier should support the predict_proba() function so that
        classification probabilities can be computed and co-training can be
        performed effectively.

    estimator2 : classifier object, (default=sklearn GaussianNB)
        The classifier object which will be trained on view 2 of the data.
        Does not need to be of the same type as ``estimator1``, but should
        support predict_proba().

    p : int, optional (default=None)
        The number of positive classifications from the unlabeled_pool
        training set which will be given a positive "label". If None, the
        default is the floor of the ratio of positive to negative examples
        in the labeled training data (at least 1). If only one of ``p`` or
        ``n`` is not None, the other will be set to be the same. When the
        labels are 0 or 1, positive is defined as 1, and in general, positive
        is the larger label.

    n : int, optional (default=None)
        The number of negative classifications from the unlabeled_pool
        training set which will be given a negative "label". If None, the
        default is the floor of the ratio of positive to negative examples
        in the labeled training data (at least 1). If only one of ``p`` or
        ``n`` is not None, the other will be set to be the same. When the
        labels are 0 or 1, negative is defined as 0, and in general, negative
        is the smaller label.

    unlabeled_pool_size : int, optional (default=75)
        The number of unlabeled_pool samples which will be kept in a
        separate pool for classification and selection by the updated
        classifier at each training iteration.

    num_iter : int, optional (default=50)
        The maximum number of training iterations to run.

    random_state : int (default=None)
        The starting random seed for fit() and class operations, passed to
        numpy.random.seed().

    Attributes
    ----------
    estimator1 : classifier object
        The classifier used on view 1.

    estimator2 : classifier object
        The classifier used on view 2.

    class_name: string
        The name of the class.

    n_views_ : int
        The number of views supported by the multi-view classifier

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of unique classes.

    p_ : int
        Number of positive examples (second label in classes_) to pull from
        unlabeled_pool and give "label" at each training round. When the
        labels are 0 or 1, positive is defined as 1, and in general, positive
        is the larger label.

    n_ : int
        Number of negative examples (second label in classes_) to pull from
        unlabeled_pool and give "label" at each training round. when the
        labels are 0 or 1, negative is defined as 0, and in general, negative
        is the smaller label.

    unlabeled_pool_size_ : int
        Size of pool of unlabeled_pool examples to classify at each iteration.

    num_iter_ : int
        Maximum number of training iterations to run.

    random_state : int (default=None)
        The starting random seed for fit() and class operations, passed to
        numpy.random.seed().

    Notes
    -----
    Multi-view co-training is most helpful for tasks in semi-supervised where
    each view offers unique information not seen in the other. As is shown in
    the example notebooks for using this algorithm, multi-view co-training
    can provide good classification results even when number of unlabeled
    samples far exceeds the number of labeled samples. The algorithm, as first
    proposed by Blum and Mitchell is the following.

    *Algorithm*

    Given:

        * a set *L* of labeled training samples (with 2 views)
        * a set *U* of unlabeled samples (with 2 views)

    Create a pool *U'* of examples by choosing *u* examples at random
    from *U*

    Loop for *k* iterations

        * Use *L* to train a classifier *h1* (``estimator1``) that considers
          only the view 1 portion of the data (i.e. Xs[0])
        * Use *L* to train a classifier *h2* (``estimator2``) that considers
          only the view 2 portion of the data (i.e. Xs[1])
        * Allow *h1* to label *p* (``self.p_``) positive and *n* (``self.n_``)
          negative samples from view 1 of *U'*
        * Allow *h2* to label *p* positive and *n* negative samples
          from view 2 of *U'*
        * Add these self-labeled samples to *L*
        * Randomly take 2*p* + 2*n* samples from *U* to replenish *U'*

    References
    ----------
    .. [#1CTC] Blum, A., & Mitchell, T. (1998, July). Combining labeled and
            unlabeled_pool data with co-training. In Proceedings of the
            eleventh annual conference on Computational learning theory
            (pp. 92-100). ACM.

    """

    def __init__(
                 self,
                 estimator1=None,
                 estimator2=None,
                 p=None,
                 n=None,
                 unlabeled_pool_size=75,
                 num_iter=50,
                 random_state=None
                 ):

        # initialize a BaseCTEstimator object
        super().__init__(estimator1, estimator2, random_state)

        # if not given, set classifiers as gaussian naive bayes estimators
        if self.estimator1 is None:
            self.estimator1 = GaussianNB()
        if self.estimator2 is None:
            self.estimator2 = GaussianNB()

        # If only 1 of p or n is not None, set them equal
        if (p is not None and n is None):
            n = p
            self.p_, self.n_ = p, n
        elif (p is None and n is not None):
            p = n
            self.p_, self.n_ = p, n
        else:
            self.p_, self.n_ = p, n

        self.n_views_ = 2  # only 2 view learning supported currently
        self.class_name = "CTClassifier"
        self.unlabeled_pool_size_ = unlabeled_pool_size
        self.num_iter_ = num_iter

        self._check_params()

    def _check_params(self):
        r"""
        Checks that cotraining parameters are valid. Throws AttributeError
        if estimators are invalid. Throws ValueError if any other parameters
        are not valid. The checks performed are:
            - estimator1 and estimator2 have predict_proba methods
            - p and n are both positive
            - unlabeled_pool_size is positive
            - num_iter is positive
        """

        # verify that estimator1 and estimator2 have predict_proba
        if (not hasattr(self.estimator1, 'predict_proba') or
                not hasattr(self.estimator2, 'predict_proba')):
            raise AttributeError("Co-training classifier must be initialized "
                                 "with classifiers supporting "
                                 "predict_proba().")

        if (self.p_ is not None and self.p_ <= 0) or (self.n_ is not None and
                                                      self.n_ <= 0):
            raise ValueError("Both p and n must be positive.")

        if self.unlabeled_pool_size_ <= 0:
            raise ValueError("unlabeled_pool_size must be positive.")

        if self.num_iter_ <= 0:
            raise ValueError("num_iter must be positive.")

    def fit(
            self,
            Xs,
            y
            ):
        r"""
        Fit the classifier object to the data in Xs, y.

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

        # verify Xs and y
        Xs, y = check_Xs_y_nan_allowed(Xs,
                                       y,
                                       multiview=True,
                                       enforce_views=self.n_views_,
                                       num_classes=2)

        y = np.array(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = list(set(y[~np.isnan(y)]))
        self.n_classes_ = len(self.classes_)

        # if both p & n are none, set as ratio of one class to the other
        if (self.p_ is None and self.n_ is None):
            num_class_n = sum(1 for y_n in y if y_n == self.classes_[0])
            num_class_p = sum(1 for y_p in y if y_p == self.classes_[1])
            p_over_n_ratio = num_class_p // num_class_n
            if p_over_n_ratio > 1:
                self.p_, self.n_ = p_over_n_ratio, 1
            else:
                self.n_, self.p_ = num_class_n // num_class_p, 1

        # extract the multiple views given
        X1 = Xs[0]
        X2 = Xs[1]

        # the full set of unlabeled samples
        U = [i for i, y_i in enumerate(y) if np.isnan(y_i)]

        # shuffle unlabeled_pool data for easy random access
        np.random.shuffle(U)

        # the small pool of unlabled samples to draw from in training
        unlabeled_pool = U[-min(len(U), self.unlabeled_pool_size_):]

        # the labeled samples
        L = [i for i, y_i in enumerate(y) if ~np.isnan(y_i)]

        # remove the pool from overall unlabeled data
        U = U[:-len(unlabeled_pool)]

        # number of rounds of co-training
        it = 0

        # machine epsilon
        eps = np.finfo(float).eps

        while it < self.num_iter_ and U:
            it += 1

            # fit each model to its respective view
            self.estimator1.fit(X1[L], y[L])
            self.estimator2.fit(X2[L], y[L])

            # predict log probability for greater spread in confidence

            y1_prob = np.log(self.estimator1.
                             predict_proba(X1[unlabeled_pool]) + eps)
            y2_prob = np.log(self.estimator2.
                             predict_proba(X2[unlabeled_pool]) + eps)

            n, p = [], []
            accurate_guesses_estimator1 = 0
            accurate_guesses_estimator2 = 0
            wrong_guesses_estimator1 = 0
            wrong_guesses_estimator2 = 0

            # take the most confident labeled examples from the
            # unlabeled pool in each category and put them in L
            for i in (y1_prob[:, 0].argsort())[-self.n_:]:
                if y1_prob[i, 0] > np.log(0.5):
                    n.append(i)
            for i in (y1_prob[:, 1].argsort())[-self.p_:]:
                if y1_prob[i, 1] > np.log(0.5):
                    p.append(i)
            for i in (y2_prob[:, 0].argsort())[-self.n_:]:
                if y2_prob[i, 0] > np.log(0.5):
                    n.append(i)
            for i in (y2_prob[:, 1].argsort())[-self.p_:]:
                if y2_prob[i, 1] > np.log(0.5):
                    p.append(i)

            # create new labels for new additions to the labeled group
            y[[unlabeled_pool[x] for x in n]] = self.classes_[0]
            y[[unlabeled_pool[x] for x in p]] = self.classes_[1]
            L.extend([unlabeled_pool[x] for x in p])
            L.extend([unlabeled_pool[x] for x in n])

            # remove newly labeled samples from unlabeled_pool
            unlabeled_pool = [elem for elem in unlabeled_pool
                              if not (elem in p or elem in n)]

            # add new elements to unlabeled_pool
            add_counter = 0
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and U:
                add_counter += 1
                unlabeled_pool.append(U.pop())

        # fit the overall model on fully "labeled" data
        self.estimator1.fit(X1[L], y[L])
        self.estimator2.fit(X2[L], y[L])

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
            The predicted class of each input example. If the two classifiers
            don't agree, pick the one with the highest predicted probability
            from predict_proba()

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.model_selection import train_test_split
        >>> from mvlearn.cotraining.ctclassifier import CTClassifier
        >>> from mvlearn.datasets.base import load_UCImultifeature
        >>> data, labels = load_UCImultifeature(select_labeled=[0,1])
        >>> X1, X2 = data[0], data[1]  # Use the first 2 views
        >>> X1_train, X1_test, l_train, l_test = train_test_split(X1, labels)
        >>> X2_train, X2_test, _, _ = train_test_split(X2, labels)
        >>> remove_idx = np.random.rand(len(l_train),) < 0.97
        >>> l_train[remove_idx] = np.nan  # simulate semi-supervised
        >>> label_ratio = len(np.where(remove_idx==False)) / len(l_train)
        >>> print('%.3f' % label_ratio)  # check labeled data proportion
        0.030
        >>> ctc = CTClassifier()
        >>> ctc.fit([X1_train, X2_train], l_train)
        >>> y_pred = ctc.predict([X1_test, X2_test])
        >>> print(y_pred[:10])  # first 10 predictions
        [0. 0. 1. 0. 1. 1. 0. 0. 0. 0.]
        """

        Xs = check_Xs(Xs,
                      multiview=True,
                      enforce_views=self.n_views_)

        X1 = Xs[0]
        X2 = Xs[1]

        # predict each view independently
        y1 = self.estimator1.predict(X1)
        y2 = self.estimator2.predict(X2)

        # initialize
        y_pred = np.zeros(X1.shape[0],)
        num_disagree = 0
        num_agree = 0

        # predict samples based on trained classifiers
        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            # if classifiers agree, use their prediction
            if y1_i == y2_i:
                y_pred[i] = y1_i
            # if classifiers don't agree, take the more confident
            else:
                y1_probs = self.estimator1.predict_proba([X1[i]])[0]
                y2_probs = self.estimator2.predict_proba([X2[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in
                               zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = self.classes_[sum_y_probs.index(max_sum_prob)]

        return y_pred

    def predict_proba(self, Xs):
        r"""
        Predict the probability of each example belonging to a each class.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of the different views of data to predict.

        Returns
        -------
        y_proba : array-like (n_samples, n_classes)
            The probability of each sample being in each class.
        """

        Xs = check_Xs(Xs,
                      multiview=True,
                      enforce_views=self.n_views_)

        X1 = Xs[0]
        X2 = Xs[1]

        y_proba = np.full((X1.shape[0], self.n_classes_), -1)
        # predict each probability independently
        y1_proba = self.estimator1.predict_proba(X1)
        y2_proba = self.estimator2.predict_proba(X2)
        # return the average probability for the sample
        return (y1_proba + y2_proba) * .5
