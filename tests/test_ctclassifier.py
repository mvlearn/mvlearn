# Gavin Mischler
# 12/12/2019

import pytest
import numpy as np
from mvlearn.cotraining.ctclassifier import CTClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier

@pytest.fixture(scope='module')
def data():
    random_seed = 10
    N = 100
    D1 = 10
    D2 = 6
    N_test = 5
    random_data = []
    np.random.seed(random_seed)
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

    return {'N_test' : N_test, 'clf_test' : clf_test,
    'random_data' : random_data, 'random_labels' : random_labels,
    'random_test' : random_test, 'random_seed' : random_seed}

def test_fit_one_class(data):
    random_labels_bad = np.floor(np.random.rand(100,))
    random_labels_bad[:-10] = np.nan
    data['clf_test'].fit(data['random_data'], random_labels_bad)

'''
EXCEPTION TESTING
'''

def test_no_predict_proba_attribute():
    with pytest.raises(AttributeError):
        clf = CTClassifier(RidgeClassifier(), RidgeClassifier())

def test_no_wrong_view_number(data):
    with pytest.raises(ValueError):
        Xs = []
        for i in range(5):
            Xs.append(np.zeros(10))

        data['clf_test'].fit(Xs, data['random_labels'])

def test_fit_incompatible_views(data):
    X1 = np.ones((100,10))
    X2 = np.zeros((99,10))
    with pytest.raises(ValueError):
        data['clf_test'].fit([X1, X2], data['random_labels'])

def test_fit_over_two_classes(data):
    with pytest.raises(ValueError):
        random_labels_bad = np.floor(2*np.random.rand(100,)+2)
        random_labels_bad[:-10] = np.nan
        random_labels_bad[0] = 10
        data['clf_test'].fit(data['random_data'], random_labels_bad)

def test_fit_zero_classes(data):
    with pytest.raises(ValueError):
        random_labels_bad = np.zeros((100,))
        random_labels_bad[:] = np.nan
        data['clf_test'].fit(data['random_data'], random_labels_bad)

def test_predict_incompatible_views(data):
    X1 = np.ones((100,10))
    X2 = np.zeros((99,10))
    with pytest.raises(ValueError):
        data['clf_test'].predict([X1, X2])

def test_predict_over_two_classes(data):
    data['clf_test'].fit(data['random_data'], data['random_labels'])
    bad_data = []
    for i in range(3):
        bad_data.append(np.ones((100,10)))
    with pytest.raises(ValueError):
        random_labels_bad = np.ones((100,))
        data['clf_test'].predict(bad_data)

def test_predict_one_class(data):
    data['clf_test'].fit(data['random_data'], data['random_labels'])
    bad_data = []
    bad_data.append(np.zeros((100,10)))
    with pytest.raises(ValueError):
        random_labels_bad = np.zeros(100)
        data['clf_test'].predict(bad_data)

def test_predict_zero_classes(data):
    data['clf_test'].fit(data['random_data'], data['random_labels'])
    bad_data = []
    with pytest.raises(ValueError):
        random_labels_bad = np.zeros(100)
        data['clf_test'].predict(bad_data)

def test_zero_p():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), p=0)

def test_negative_p():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), p=-1)

def test_zero_n():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), n=0)

def test_negative_n():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), n=-1)

def test_zero_pool_size():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), unlabeled_pool_size=0)

def test_negative_pool_size():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), unlabeled_pool_size=-1)

def test_zero_num_iter():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), num_iter=0)

def test_negative_num_iter():
    with pytest.raises(ValueError):
        clf = CTClassifier(GaussianNB(), GaussianNB(), num_iter=-1)

'''
FUNCTION TESTING
'''

def test_fit_defaultclassifier(data):
    clf = CTClassifier(random_state=10)
    clf.fit(data['random_data'], data['random_labels'])
    y_pred_test = clf.predict(data['random_test'])
    y_pred_prob = clf.predict_proba(data['random_test'])

    truth = [2., 3., 2., 2., 2.]
    truth_proba = [[0.88555037, 0.11444963],
    [0.05650123, 0.94349877],
    [0.50057741, 0.49942259],
    [0.89236186, 0.10763814],
    [0.95357416, 0.04642584]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001

def test_predict_default(data):
    data['clf_test'].fit(data['random_data'], data['random_labels'])
    y_pred_test = data['clf_test'].predict(data['random_test'])
    y_pred_prob = data['clf_test'].predict_proba(data['random_test'])

    truth = [2., 3., 2., 2., 2.]
    truth_proba = [[0.88555037, 0.11444963],
    [0.05650123, 0.94349877],
    [0.50057741, 0.49942259],
    [0.89236186, 0.10763814],
    [0.95357416, 0.04642584]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001

def test_predict_check_p_n(data):
    labels1 = np.zeros(100,)
    labels1[:5] = 4 # 5 "negative"
    labels1[5:15] = 6 # 10 "positive"
    labels1[15:] = np.nan
    clf = CTClassifier(random_state=0)
    clf.fit(data['random_data'], labels1)
    assert clf.p_ == 2
    assert clf.n_ == 1

    labels2 = np.zeros(100,)
    labels2[:5] = 6 # 5 "positive"
    labels2[5:15] = 4 # 10 "negative"
    labels2[15:] = np.nan
    clf = CTClassifier(random_state=0)
    clf.fit(data['random_data'], labels2)
    assert clf.p_ == 1
    assert clf.n_ == 2

    labels1 = np.zeros(100,)
    labels1[:5] = 4 # 5 "negative"
    labels1[5:15] = 6 # 10 "positive"
    labels1[15:] = np.nan
    clf = CTClassifier(p=4, n=3, random_state=0)
    clf.fit(data['random_data'], labels1)
    assert clf.p_ == 4
    assert clf.n_ == 3

def test_predict_set_p(data):
    random_seed = 10
    gnb1 = GaussianNB()
    gnb2 = GaussianNB()
    clf = CTClassifier(gnb1, gnb2, p=12, random_state=random_seed)
    clf.fit(data['random_data'], data['random_labels'])
    y_pred_test = clf.predict(data['random_test'])
    y_pred_prob = clf.predict_proba(data['random_test'])

    truth = [3., 3., 3., 3., 3.]
    truth_proba = [[0.31422418, 0.68577582],
    [0.40938282, 0.59061718],
    [0.48448605, 0.51551395],
    [0.38853225, 0.61146775],
    [0.22972488, 0.77027512]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001

def test_predict_set_n(data):
    random_seed = 10
    gnb1 = GaussianNB()
    gnb2 = GaussianNB()
    clf = CTClassifier(gnb1, gnb2, n=9, random_state=random_seed)
    clf.fit(data['random_data'], data['random_labels'])
    y_pred_test = clf.predict(data['random_test'])
    y_pred_prob = clf.predict_proba(data['random_test'])

    truth = [3., 3., 2., 3., 3.]
    truth_proba = [[0.29020704, 0.70979296],
    [0.44024614, 0.55975386],
    [0.5710383 , 0.4289617 ],
    [0.37366059, 0.62633941],
    [0.22157484, 0.77842516]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001

def test_predict_unlabeled_pool_size(data):
    random_seed = 10
    gnb1 = GaussianNB()
    gnb2 = GaussianNB()
    clf = CTClassifier(gnb1, gnb2, unlabeled_pool_size=20, random_state=random_seed)
    clf.fit(data['random_data'], data['random_labels'])
    y_pred_test = clf.predict(data['random_test'])
    y_pred_prob = clf.predict_proba(data['random_test'])

    truth = [2., 3., 2., 2., 2.]
    truth_proba = [[0.55708013, 0.44291987],
    [0.29591617, 0.70408383],
    [0.50441055, 0.49558945],
    [0.99276393, 0.00723607],
    [0.95221514, 0.04778486]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001

def test_predict_num_iter(data):
    random_seed = 10
    gnb1 = GaussianNB()
    gnb2 = GaussianNB()
    clf = CTClassifier(gnb1, gnb2, num_iter=9, random_state=random_seed)
    clf.fit(data['random_data'], data['random_labels'])
    y_pred_test = clf.predict(data['random_test'])
    y_pred_prob = clf.predict_proba(data['random_test'])

    truth = [2., 3., 2., 2., 2.]
    truth_proba = [[0.88555037, 0.11444963],
    [0.05650123, 0.94349877],
    [0.50057741, 0.49942259],
    [0.89236186, 0.10763814],
    [0.95357416, 0.04642584]]

    for i in range(data['N_test']):
        assert y_pred_test[i] == truth[i]

    for i in range(data['N_test']):
        for j in range(2):
            assert abs(y_pred_prob[i,j] - truth_proba[i][j]) < 0.00000001
