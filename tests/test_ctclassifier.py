import pytest
import numpy as np
from multiview.predict import CTClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


def test_no_predict_proba_attribute():
	with pytest.raises(AttributeError):
		clf = CTClassifier(LinearSVC(), LinearSVC())

def test_predict():
	random_seed = 10
	N = 100
	D1 = 10
	D2 = 6
	N_test = 5
	random_data = []
	random_data.append(np.random.rand(N,D1))
	random_data.append(np.random.rand(N,D2))
	random_labels = np.floor(2*np.random.rand(N,)+2)
	random_labels[:-10] = np.nan
	random_test = []
	random_test.append(np.random.rand(N_test, D1))
	random_test.append(np.random.rand(N_test, D2))
	gnb1 = GaussianNB()
	gnb2 = GaussianNB()
	clf_test = CTClassifier(gnb1, gnb2, random_state=random_seed)
	clf_test.fit(random_data, random_labels)
	y_pred_test = clf_test.predict(random_test)
	y_pred_prob = clf_test.predict_proba(random_test)
	truth = [3,3,3,3,2]
	for i in range(N_test):
	    assert y_pred_test[i] == truth[i]

	truth_proba = [[0.0308463, 0.9691537],
               [0.0644134, 0.9355866],
               [0.00123236, 0.99876764],
               [0.48250598, 0.51749402],
               [0.56829824, 0.43170176]]
	for i in range(N_test):
	    for j in range(2):
	        assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.000001





