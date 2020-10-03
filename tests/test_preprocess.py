import pytest
import numpy as np
from mvlearn.preprocessing import train_test_split

RANDOM_SEED = 10
np.random.seed(RANDOM_SEED)

@pytest.fixture(scope='module')
def r_data():
    data = list()
    for _ in range(3):
        data.append(np.random.random((20, 10)))

    labels = np.random.random((20,))
        
    return data, labels

# Exception testing
def test_Xs_not_valid():
    with pytest.raises(ValueError):
        data = 10
        splits = train_test_split(data, random_state=RANDOM_SEED)

def test_y_not_valid(r_data):
    with pytest.raises(ValueError):
        labels = 4
        splits = train_test_split(r_data[0], labels, random_state=RANDOM_SEED)

        
# Function testing

def test_split(r_data):
    train_Xs, test_Xs = train_test_split(r_data[0], random_state=RANDOM_SEED)
    assert len(train_Xs) == 3
    assert len(test_Xs) == 3

    for X in train_Xs:
        assert X.shape == (15, 10)
    for X in test_Xs:
        assert X.shape == (5, 10)

def test_split_params(r_data):
    
    train_Xs, test_Xs = train_test_split(r_data[0], test_size=0.2,
                                        train_size=None, random_state=RANDOM_SEED,
                                        shuffle=True, stratify=None)
    assert len(train_Xs) == 3
    assert len(test_Xs) == 3

    for X in train_Xs:
        assert X.shape == (16, 10)
    for X in test_Xs:
        assert X.shape == (4, 10)

def test_split_y(r_data):
    train_Xs, test_Xs, train_y, test_y = train_test_split(r_data[0], r_data[1],
                                                          random_state=RANDOM_SEED)
    assert len(train_Xs) == 3
    assert len(test_Xs) == 3
    
    for X in train_Xs:
        assert X.shape == (15, 10)
    for X in test_Xs:
        assert X.shape == (5, 10)

    assert train_y.shape == (15,)
    assert test_y.shape == (5,)

        
def test_data_split_params_y(r_data):
    
    splits = train_test_split(r_data[0], r_data[1],
                              test_size=0.2, train_size=None,
                              random_state=RANDOM_SEED, shuffle=True,
                              stratify=None)
    train_Xs, test_Xs, train_y, test_y = tuple(splits)
    
    assert len(train_Xs) == 3
    assert len(test_Xs) == 3
    
    for X in train_Xs:
        assert X.shape == (16, 10)
    for X in test_Xs:
        assert X.shape == (4, 10)

    assert train_y.shape == (16,)
    assert test_y.shape == (4,)
