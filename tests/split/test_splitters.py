import numpy as np
from numpy.testing import assert_raises

import pytest

from mvlearn.split import ConcatSplitter


@pytest.mark.parametrize(
    "n_features",
    (
        [
            2,
        ],
        [2, 3],
        [2, 3, 4],
    ),
)
def test_split(n_features):
    rng = np.random.RandomState(0)
    n_samples = 10
    X = rng.randn(n_samples, np.sum(n_features))
    st = ConcatSplitter(n_features)
    st.fit(X)
    Xs_transformed = st.transform(X)
    # Check dimensions
    for X_, n_feature in zip(Xs_transformed, n_features):
        assert X_.shape == (n_samples, n_feature)
    assert st.n_total_features_ == sum(n_features)
    assert st.n_views_ == len(n_features)
    # Check fit transform
    Xs_transformed2 = st.fit_transform(X)
    for X1, X2 in zip(Xs_transformed, Xs_transformed2):
        assert (X1 == X2).all()
    # Back transform
    X_init = st.inverse_transform(Xs_transformed)
    assert X_init.shape == X.shape
    assert (X_init == X).all()


def test_errors():
    X = np.random.randn(10, 5)
    with assert_raises(ValueError):
        ConcatSplitter([2, 2]).fit(X)
    st = ConcatSplitter([2, 3]).fit(X)
    Xs = [np.random.randn(10, n) for n in [2, 2]]
    with assert_raises(ValueError):
        st.inverse_transform(Xs)
