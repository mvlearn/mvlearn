# Gavin Mischler
# 10/21/2019

import pytest
from mvlearn.cotraining.base import BaseCoTrainEstimator
from sklearn.base import BaseEstimator

"""
BASE CLASS TESTING
"""

def test_base_ctclassifier():
    base_clf = BaseCoTrainEstimator()
    assert isinstance(base_clf, BaseEstimator)
    base_clf.fit([],[])
    base_clf.predict([])
