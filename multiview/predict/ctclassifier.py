import numpy as np
from sklearn.naive_bayes import GaussianNB

class CTClassifier(object):
    """
    Co-Training Classifier

    This class implements the co-training classifier similar to as described
    in [1]. This should ideally be used on 2 views of the input data which
    satisfy the 3 conditions for multi-view co-training (sufficiency,
    compatibility, conditional independence) as detailed in [1].


    Parameters
    ----------
    h1 : classifier object
        The classifier object which will be trained on view 1 of the data.
        This classifier should support the predict_proba() function so that
        classification probabilities can be computed and co-training can be
        performed effectively.

    h2 : classifier object
        The classifier object which will be trained on view 2 of the data.
        Does not need to be of the same type as h1, but should support
        predict_proba().

    Attributes
    ----------
    h1 : classifier object
        The classifier used on view 1.

    h2 : classifier object
        The classifier used on view 2.

    n_views_ : int
        The number of views supported by the multi-view classifier

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of unique classes.

    p_ : int
        Number of positive examples (second label in classes_) to pull from
        unlabeled_pool and give "label" at each training round.

    n_ : int
        Number of positive examples (second label in classes_) to pull from
        unlabeled_pool and give "label" at each training round.

    unlabeled_pool_pool_size_ : int
        Size of pool of unlabeled_pool examples to classify at each iteration.

    num_iter_ : int
        Maximum number of training iterations to run.


    References
    ----------
    [1] Blum, A., & Mitchell, T. (1998, July). Combining labeled and unlabeled_pool
        data with co-training. In Proceedings of the eleventh annual
        conference on Computational learning theory (pp. 92-100). ACM.

    """


    def __init__(self, h1=None, h2=None, random_state=0):

        # if not given, set as gaussian naive bayes estimators
        if h1 is None:
            h1 = GaussianNB()
        if h2 is None:
            h2 = GaussianNB()


        # verify that h1 and h2 have predict_proba
        if (not hasattr(h1, 'predict_proba') or not hasattr(h2, 'predict_proba'
            )):
            raise AttributeError("Co-training classifier must be initialized "
                "with classifiers supporting the predict_proba() function.")

        self.h1 = h1
        self.h2 = h2
        self.n_views_ = 2 # only 2 view learning supported

        self.random_state = random_state

        # for testing
        self.partial_error_ = []
        # for testing with training data
        self.partial_train_error_ = []
        self.class_name = "CTClassifier"


    def fit(self, Xs, y, p=None, n=None, unlabeled_pool_pool_size=75, num_iter=50, y_train_full=None, X1_test=None, X2_test=None, y_test=None):
        """
        Fit the classifier object to the data in Xs, y.

        Parameters
        ----------
        Xs : list of numpy arrays (each must have same first dimension)
            The list should be length 2 (since only 2 view data is currently
            supported for co-training). View 1 (X1) is the first element in
            the list and should have shape (n_samples, n1_features). View 2 (X2)
            is the second element in the list and should have shape (n-samples,
            n2_features)

        y : array-like of shape (n_samples,)
            The labels of the training data. unlabeled_pool examples should have
            label np.nan.

        p : int, optional (default=None)
            The number of positive classifications from the unlabeled_pool
            training set which will be given a positive "label". If None, the
            default is the floor of the ratio of positive to negative examples
            in the labeled training data (at least 1). If only one of p or n
            is not None, the other will be set to be the same.

        n : int, optional (default=None)
            The number of negative classifications from the unlabeled_pool
            training set which will be given a negative "label". If None, the
            default is the floor of the ratio of positive to negative examples
            in the labeled training data (at least 1). If only one of p or n
            is not None, the other will be set to be the same.

        unlabeled_pool_pool_size : int, optional (default=75)
            The number of unlabeled_pool samples which will be kept in a separate pool
            for classification and selection by the updated classifier at each
            training iteration.

        num_iter : int, optional (default=50)
            The maximum number of training iterations to run.

        """
        #TODO: input validation
        #TODO: make sure classifiers agree 

        if len(Xs) != self.n_views_:
            raise ValueError("{0:s} must provide {1:d} views; got {2:d} views"
                             .format(self.class_name, self.n_views_,
                                len(Xs)))

        X1 = Xs[0]
        X2 = Xs[1]

        y = np.array(y)

        np.random.seed(self.random_state)


        # if not exactly 2 classes, raise error
        self.classes_ = set(y[~np.isnan(y)])
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ > 2:
            raise ValueError("{0:s} supports only binary classification. "
                             "y contains {1:d} classes"
                             .format(self.class_name, self.n_classes_))
        if self.n_classes_ == 1:
            raise ValueError("{0:s} requires 2 classes; got 1 class"
                             .format(self.class_name))
        if self.n_classes_ == 0:
            raise ValueError("Insufficient labeled data")


        # If only 1 of p or n is not None, set them equal
        if (p is not None and n is None):
            n = p
        elif (p is None and n is not None):
            p = n
        elif (p is None and n is None):
            num_class_n = sum(1 for y_n in y if y_n == self.classes_[0])
            num_class_p = sum(1 for y_p in y if y_p == self.classes_[1])
            p_over_n_ratio = num_class_p // num_class_n
            if p_over_n_ratio > 1:
                self.p_ = p_over_n_ratio
                self.n_ = 1
            else:
                self.n_ = num_class_n // num_class_p
                self.p_ = 1
        else:
            self.p_ = p
            self.n_ = n


        self.unlabeled_pool_pool_size_ = unlabeled_pool_pool_size
        self.num_iter_ = num_iter

        print(self.n_)
        print(self.p_)

        # the set of unlabeled_pool samples
        U = [i for i, y_i in enumerate(y) if np.isnan(y_i)]

        # shuffle unlabeled_pool data for easy random access    
        np.random.shuffle(U)

        unlabeled_pool = U[-min(len(U), self.unlabeled_pool_pool_size_):]

        # labeled samples
        L = [i for i, y_i in enumerate(y) if ~np.isnan(y_i)]

        # remove the pool from overall unlabeled data
        U = U[:-len(unlabeled_pool)]

        # rounds of co-training
        it = 0

        while it < self.num_iter_ and U:
            it += 1

            self.h1.fit(X1[L], y[L])
            self.h2.fit(X2[L], y[L])

            y1_prob = self.h1.predict_log_proba(X1[unlabeled_pool])
            y2_prob = self.h2.predict_log_proba(X2[unlabeled_pool])

            n, p = [], []
            accurate_guesses_h1 = 0
            accurate_guesses_h2 = 0
            wrong_guesses_h1 = 0
            wrong_guesses_h2 = 0


            for i in (y1_prob[:,0].argsort())[-self.n_:]:
                if y1_prob[i,0] > np.log(0.5):
                    n.append(i)

            for i in (y1_prob[:,1].argsort())[-self.p_:]:
                if y1_prob[i,1] > np.log(0.5):
                    p.append(i)

            for i in (y2_prob[:,0].argsort())[-self.n_:]:
                if y2_prob[i,0] > np.log(0.5):
                    n.append(i)

            for i in (y2_prob[:,1].argsort())[-self.p_:]:
                if y2_prob[i,1] > np.log(0.5):
                    p.append(i)

            # create new labels for new additions to the labeled group
            y[[unlabeled_pool[x] for x in n]] = self.classes_[0]
            y[[unlabeled_pool[x] for x in p]] = self.classes_[1]
            L.extend([unlabeled_pool[x] for x in p])
            L.extend([unlabeled_pool[x] for x in n])

            # remove newly labeled samples from unlabeled_pool
            unlabeled_pool = [elem for elem in unlabeled_pool if not (elem in p or elem in n)]

            #add new elements to unlabeled_pool
            add_counter = 0 #number we have added from U to unlabeled_pool
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and U:
                add_counter += 1
                unlabeled_pool.append(U.pop())


            # if input testing data as well, find the incrememtal update on accuracy
            if X1_test is not None and X2_test is not None and y_test is not None:
                y_pred = self.predict([X1_test, X2_test])
                self.partial_error_.append(1-accuracy_score(y_test, y_pred))
                y_pred = self.predict([X1, X2])
                self.partial_train_error_.append(1-accuracy_score(y_train_full, y_pred))

        # fit the overall model on fully "labeled" data
        self.h1.fit(X1[L], y[L])
        self.h2.fit(X2[L], y[L])

        return (self.partial_train_error_, self.partial_error_)


    def predict(self, Xs):
        """
        Predict the classes of the examples in the two input views.

        Parameters
        ----------
        Xs : list of numpy arrays (each with the same first dimension)
            The list should be length 2 (since only 2 view data is currently
            supported for co-training). View 1 (X1) is the first element in
            the list and should have shape (n_samples, n1_features). View 2 (X2)
            is the second element in the list and should have shape (n-samples,
            n2_features)


        Returns
        -------
        y_pred : array-like (n_samples,)
            The predicted class of each input example. If the two classifiers
            don't agree, pick the one with the highest predicted probability
            from predict_proba()

        """

        if len(Xs) != self.n_views_:
            raise ValueError("{0:s} must provide {1:d} views; got {2:d} views"
                             .format(self.class_name, self.n_views_,
                                len(Xs)))
        X1 = Xs[0]
        X2 = Xs[1]

        if X1.shape[0] != X2.shape[0]:
            raise ValueError("2 provided views have incompatible dimensions, "
                             " they must have the same number of samples.")

        y1 = self.h1.predict(X1)
        y2 = self.h2.predict(X2)

        #fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
        y_pred = np.zeros(X1.shape[0],)
        #y_pred = np.asarray([-1] * X1.shape[0])
        num_disagree = 0
        num_agree = 0

        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
                num_agree += 1
            else:
                y1_probs = self.h1.predict_proba([X1[i]])[0]
                y2_probs = self.h2.predict_proba([X2[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = self.classes_[sum_y_probs.index(max_sum_prob)]
                num_disagree += 1

        return y_pred

    def predict_proba(self, Xs):
        """
        Predict the probability of each example belonging to a each class.

        Parameters
        ----------
        Xs : list of numpy arrays (each with the same first dimension)
            The list should be length 2 (since only 2 view data is currently
            supported for co-training). View 1 (X1) is the first element in
            the list and should have shape (n_samples, n1_features). View 2 (X2)
            is the second element in the list and should have shape (n-samples,
            n2_features)


        Returns
        -------
        y_proba : array-like (n_samples, n_classes)
            The probability of each sample being in each class.

        """

        if len(Xs) != self.n_views_:
            raise ValueError("{0:s} must provide {1:d} views; got {2:d} views"
                             .format(self.class_name, self.n_views_,
                                len(Xs)))

        X1 = Xs[0]
        X2 = Xs[1]

        y_proba = np.full((X1.shape[0], self.n_classes_), -1)

        y1_proba = self.h1.predict_proba(X1)
        y2_proba = self.h2.predict_proba(X2)

        return (y1_proba + y2_proba) * .5

        
