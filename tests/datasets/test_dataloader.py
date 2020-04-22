# Gavin Mischler
# 10/17/2019

import pytest
import numpy as np
from mvlearn.datasets.base import load_UCImultifeature

def test_UCImultifeature_dataloader():
    # load data
    data, labels = load_UCImultifeature()

    assert len(data) == 6
    assert labels.shape[0] == 2000

    # check size of data
    for i in range(6):
        assert data[i].shape[0] == 2000

    data1, labels1 = load_UCImultifeature()

    # check data and labels are same
    assert(np.allclose(data[0], data1[0]))
    assert(np.allclose(labels, labels1))

def test_UCImultifeature_randomstate_sameordifferent():

    # load data
    data, labels = load_UCImultifeature(shuffle=True, random_state=2)
    data1, labels1 = load_UCImultifeature(shuffle=True, random_state=5)
    data2, labels2 = load_UCImultifeature(shuffle=True, random_state=2)
    data3, labels3 = load_UCImultifeature(shuffle=False)

    assert len(data) == 6
    assert labels.shape[0] == 2000

    # check size of data
    for i in range(6):
        assert data[i].shape[0] == 2000

    # check data is same
    for idx in range(6):
        assert(np.allclose(data[idx], data2[idx]))
        assert(not np.allclose(data[idx], data1[idx]))
        assert(not np.allclose(data[idx], data3[idx]))
        assert(not np.allclose(data1[idx], data3[idx]))

def test_UCImultifeature_dataloader_select():
    # load data
    lab = [0,1,2]
    data, labels = load_UCImultifeature(select_labeled=lab)

    assert len(data) == 6

    assert labels.shape[0] == 600
    labels_set = list(set(labels))
    assert len(labels_set) == len(lab)
    for j, lab_in_set in enumerate(labels_set):
        assert lab_in_set == lab[j]

    # check size of data
    for i in range(6):
        assert data[i].shape[0] == 600

def test_UCImultifeature_dataloader_badselect():
    with pytest.raises(ValueError):
        data, labels = load_UCImultifeature(select_labeled=[])

def test_UCImultifeature_dataloader_badselect2():
    long_list = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    with pytest.raises(ValueError):
        data, labels = load_UCImultifeature(select_labeled=long_list)

def test_UCImultifeature_dataloader_badselect3():
    bad_list = [0,2,4,-2]
    with pytest.raises(ValueError):
        data, labels = load_UCImultifeature(select_labeled=bad_list)
