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

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of unique classes.

    p_ : int
        Number of positive examples (second label in classes_) to pull from
        unlabeled and give "label" at each training round.

    n_ : int
        Number of positive examples (second label in classes_) to pull from
        unlabeled and give "label" at each training round.

    unlabeled_pool_size_ : int
        Size of pool of unlabeled examples to classify at each iteration.

    num_iter_ : int
        Maximum number of training iterations to run.


    References
    ----------

    [1] Blum, A., & Mitchell, T. (1998, July). Combining labeled and unlabeled
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

        np.random.seed(random_state)

        

        # for testing
        self.partial_error_ = []
        # for testing with training data
        self.partial_train_error_ = []
        self.class_name = "CTClassifier"


    def fit(self, Xs, y, p=None, n=None, unlabeled_pool_size=75, num_iter=50, y_train_full=None, X1_test=None, X2_test=None, y_test=None):
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
            The labels of the training data. Unlabeled examples should have
            label np.nan.

        p : int, optional (default=None)
            The number of positive classifications from the unlabeled
            training set which will be given a positive "label". If None, the
            default is the floor of the ratio of positive to negative examples
            in the labeled training data (at least 1). If only one of p or n
            is not None, the other will be set to be the same.

        n : int, optional (default=None)
            The number of negative classifications from the unlabeled
            training set which will be given a negative "label". If None, the
            default is the floor of the ratio of positive to negative examples
            in the labeled training data (at least 1). If only one of p or n
            is not None, the other will be set to be the same.

        unlabeled_pool_size : int, optional (default=75)
            The number of unlabeled samples which will be kept in a separate pool
            for classification and selection by the updated classifier at each
            training iteration.

        num_iter : int, optional (default=50)
            The maximum number of training iterations to run.

        """

        X1 = Xs[0]
        X2 = Xs[1]

        # convert to numpy array
        y = np.asarray(y)

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


        self.unlabeled_pool_size_ = unlabeled_pool_size
        self.num_iter_ = num_iter

        print(self.n_)
        print(self.p_)

        assert(self.p_ > 0 and self.n_ > 0 and self.num_iter_ > 0 and self.unlabeled_pool_size_ > 0)

        #the set of unlabeled samples
        U = [i for i, y_i in enumerate(y) if np.isnan(y_i)]

        #we randomize here, and then just take from the back so we don't have to sample every time
        np.random.seed(10)
        np.random.shuffle(U)

        #this is U' in paper
        unlabeled = U[-min(len(U), self.unlabeled_pool_size_):]

        #the samples that are initially labeled
        L = [i for i, y_i in enumerate(y) if ~np.isnan(y_i)]

        #remove the samples in unlabeled from U
        U = U[:-len(unlabeled)]

        it = 0 #number of cotraining iterations we've done so far

        #loop until we have assigned labels to everything in U or we hit our iteration break condition
        while it < self.num_iter_ and U:
            it += 1


            self.h1.fit(X1[L], y[L])
            self.h2.fit(X2[L], y[L])
            print(len(L))

            y1_prob = self.h1.predict_log_proba(X1[unlabeled])
            y2_prob = self.h2.predict_log_proba(X2[unlabeled])

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

            #label the samples and remove the newly added samples from unlabeled
            y[[unlabeled[x] for x in p]] = 1
            y[[unlabeled[x] for x in n]] = 0

            L.extend([unlabeled[x] for x in p])
            L.extend([unlabeled[x] for x in n])

            unlabeled = [elem for elem in unlabeled if not (elem in p or elem in n)]

            #add new elements to unlabeled
            add_counter = 0 #number we have added from U to unlabeled
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and U:
                add_counter += 1
                unlabeled.append(U.pop())


            # if input testing data as well, find the incrememtal update on accuracy
            if X1_test is not None and X2_test is not None and y_test is not None:
                y_pred = self.predict([X1_test, X2_test])
                self.partial_error_.append(1-accuracy_score(y_test, y_pred))
                y_pred = self.predict([X1, X2])
                self.partial_train_error_.append(1-accuracy_score(y_train_full, y_pred))


            #TODO: Handle the case where the classifiers fail to agree on any of the samples (i.e. both n and p are empty)


        #fit the final model
        self.h1.fit(X1[L], y[L])
        self.h2.fit(X2[L], y[L])

        return (self.partial_train_error_, self.partial_error_)


    #TODO: Move this outside of the class into a util file.
    def supports_proba(self, clf, x):
        """Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
        try:
            clf.predict_proba([x])
            return True
        except:
            return False

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

        X1 = Xs[0]
        X2 = Xs[1]

        if len(Xs) != self.n_classes_:
            raise ValueError("{0:s} must provide {1:d} classes; got classes"
                             .format(self.class_name, self.n_classes_,
                                len(Xs)))

        y1 = self.h1.predict(X1)
        y2 = self.h2.predict(X2)

        proba_supported = self.supports_proba(self.h1, X1[0]) and self.supports_proba(self.h2, X2[0])

        #fill y_pred with -1 so we can identify the samples in which the classifiers failed to agree
        y_pred = np.asarray([-1] * X1.shape[0])
        num_disagree = 0
        num_agree = 0

        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
                num_agree += 1
            elif proba_supported:
                y1_probs = self.h1.predict_proba([X1[i]])[0]
                y2_probs = self.h2.predict_proba([X2[i]])[0]
                sum_y_probs = [prob1 + prob2 for (prob1, prob2) in zip(y1_probs, y2_probs)]
                max_sum_prob = max(sum_y_probs)
                y_pred[i] = sum_y_probs.index(max_sum_prob)
                num_disagree += 1
            else:
                #the classifiers disagree and don't support probability, so we guess
                y_pred[i] = random.randint(0, 1)

        print("agree: " + str(num_agree))
        print("disagree: " + str(num_disagree))


        #check that we did everything right
        assert not (-1 in y_pred)

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

        X1 = Xs[0]
        X2 = Xs[1]

        if len(Xs) != self.n_classes_:
            raise ValueError("{0:s} must provide {1:d} classes; got classes"
                             .format(self.class_name, self.n_classes_,
                                len(Xs)))

        y_proba = np.full((X1.shape[0], self.n_classes_), -1)

        y1_proba = self.h1.predict_proba(X1)
        y2_proba = self.h2.predict_proba(X2)

        for i, (y1_i_dist, y2_i_dist) in enumerate(zip(y1_proba, y2_proba)):
            y_proba[i][0] = (y1_i_dist[0] + y2_i_dist[0]) / 2
            y_proba[i][1] = (y1_i_dist[1] + y2_i_dist[1]) / 2

        _epsilon = 0.0001
        assert all(abs(sum(y_dist) - 1) <= _epsilon for y_dist in y_proba)
        return y_proba
