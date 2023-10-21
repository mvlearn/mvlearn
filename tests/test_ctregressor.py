import pytest
import numpy as np
from mvlearn.semi_supervised.ctregression import CTRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope='module')
def data():
    random_seed = 10
    N = 100
    D1 = 10
    D2 = 6
    N_test = 5
    random_data = []
    np.random.seed(random_seed)
    random_data.append(np.random.rand(N, D1))
    random_data.append(np.random.rand(N, D2))
    random_labels = np.random.rand(N)
    random_labels[:-10] = np.nan
    random_test = []
    random_test.append(np.random.rand(N_test, D1))
    random_test.append(np.random.rand(N_test, D2))
    knn1 = KNeighborsRegressor()
    knn2 = KNeighborsRegressor()
    reg_test = CTRegressor(
        estimator1=knn1, estimator2=knn2, random_state=random_seed)

    return {
        'N_test': N_test,
        'reg_test': reg_test,
        'random_data': random_data,
        'random_labels': random_labels,
        'random_test': random_test,
        'random_seed': random_seed}


'''
EXCEPTION TESTING
'''


def test_not_knn():
    with pytest.raises(AttributeError):
        ctr = CTRegressor(
            estimator1=LinearRegression(), estimator2=LinearRegression())


def test_k_neighbors_zero():
    with pytest.raises(ValueError):
        ctr = CTRegressor(k_neighbors=0)


def test_unlabeled_pool_size_zero():
    with pytest.raises(ValueError):
        ctr = CTRegressor(unlabeled_pool_size=0)


def test_num_iter_zero():
    with pytest.raises(ValueError):
        ctr = CTRegressor(num_iter=0)


def test_k_neighbors_negative():
    with pytest.raises(ValueError):
        ctr = CTRegressor(k_neighbors=-1)


def test_unlabeled_pool_size_negative():
    with pytest.raises(ValueError):
        ctr = CTRegressor(unlabeled_pool_size=-1)


def test_num_iter_negative():
    with pytest.raises(ValueError):
        ctr = CTRegressor(num_iter=-1)


def test_no_wrong_view_number(data):
    with pytest.raises(ValueError):
        Xs = []
        for _ in range(5):
            Xs.append(np.zeros(10))
        data['reg_test'].fit(Xs, data['random_labels'])


def test_fit_incompatible_views(data):
    X1 = np.ones((100, 10))
    X2 = np.zeros((99, 10))
    with pytest.raises(ValueError):
        data['reg_test'].fit([X1, X2], data['random_labels'])


def test_fit_incompatible_label(data):
    random_labels = np.random.rand(data['random_data'][0].shape[0] + 2)
    with pytest.raises(ValueError):
        data['reg_test'].fit(data['random_data'], random_labels)


def test_predict_incompatible_views(data):
    X1 = np.ones((100, 10))
    X2 = np.zeros((99, 10))
    with pytest.raises(ValueError):
        data['reg_test'].predict([X1, X2])


def test_set_k_neighbors_less_than_labelled_size():
    X1 = [[0], [1], [2], [3], [4], [5], [6]]
    X2 = [[2], [3], [4], [6], [7], [8], [10]]
    y = [10, 11, 12, 13, 14, 15, 16]
    y_train = [10, np.nan, 12, np.nan, 14, np.nan, 16]
    ctr = CTRegressor(k_neighbors=5, random_state=42)
    with pytest.raises(ValueError):
        ctr.fit([X1, X2], y_train)


'''
FUNCTION TESTING
'''


def test_fit_regressor(data):
    truth = [0.43418396, 0.43633812, 0.52360168, 0.49169045, 0.4694159]
    ctr = CTRegressor(random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.0000001


def test_set_num_iter(data):
    truth = [0.34590772, 0.47308874, 0.43699262, 0.48092033, 0.41690168]
    num_iter = 10
    ctr = CTRegressor(num_iter=num_iter, random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert ctr.num_iter == num_iter
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001


def test_set_k_neighbors(data):
    truth = [0.48113792, 0.44530868, 0.5269744, 0.50646773, 0.53355599]
    k_neighbors = 4
    ctr = CTRegressor(k_neighbors=k_neighbors, random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert ctr.k_neighbors == k_neighbors
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001


def test_set_unlabeled_pool_size_as_one(data):
    truth = [0.39130458, 0.48810745, 0.45749297, 0.48121387, 0.49026534]
    unlabeled_pool_size = 1
    ctr = CTRegressor(unlabeled_pool_size=unlabeled_pool_size, random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert ctr.unlabeled_pool_size == unlabeled_pool_size
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001


def test_set_unlabeled_pool_size_as_two(data):
    truth = [0.45287164, 0.44940644, 0.45653891, 0.48608956, 0.53500596]
    unlabeled_pool_size = 2
    ctr = CTRegressor(unlabeled_pool_size=unlabeled_pool_size, random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert ctr.unlabeled_pool_size == unlabeled_pool_size
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001


def test_set_unlabeled_pool_size_as_ten(data):
    truth = [0.48274064, 0.43065151, 0.45900958, 0.50178144, 0.49710094]
    unlabeled_pool_size = 10
    ctr = CTRegressor(unlabeled_pool_size=unlabeled_pool_size, random_state=10)
    ctr.fit(data['random_data'], data['random_labels'])
    pred = (ctr.predict(data['random_test'])).tolist()
    assert ctr.unlabeled_pool_size == unlabeled_pool_size
    assert len(truth) == len(pred)
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001


def test_set_n_neighbors_as_one():
    X1 = [[0], [1], [2], [3], [4], [5], [6]]
    X2 = [[2], [3], [4], [6], [7], [8], [10]]
    y = [10, -200, 12, 13, -100, 15, 16]
    y_train = [10, np.nan, np.nan, 13, np.nan, 15, 16]
    truth = [10.75, 10.75, 12.25, 13.75, 13.75, 14.75, 15.5]
    ctr = CTRegressor(
        KNeighborsRegressor(n_neighbors=2),
        KNeighborsRegressor(n_neighbors=2),
        k_neighbors=2, random_state=42)
    ctr.fit([X1, X2], y_train)
    pred = ctr.predict([X1, X2])
    for i, j in zip(truth, pred):
        assert abs(i-j) < 0.00000001
